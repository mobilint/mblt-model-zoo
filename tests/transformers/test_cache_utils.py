"""Unit tests for Mobilint cache snapshot helpers."""

from __future__ import annotations

import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.cache_utils import (
    MobilintBeamCache,
    MobilintCache,
    MobilintDeepStackCache,
    MobilintEagle3Cache,
    MobilintWhisperCache,
    append_whisper_beam_debug_event,
    build_mobilint_cache_from_model,
    is_whisper_beam_debug_trace_enabled,
)


class _FakeMxqModel:
    """Minimal MXQ stub for cache snapshot tests."""

    def __init__(self) -> None:
        self.loaded: list[tuple[int, list[bytes]]] = []

    def dump_cache_memory(self, cache_id: int) -> list[bytes]:
        """Return a stable in-memory cache payload for the requested cache id."""
        return [f"cache-{cache_id}".encode("utf-8")]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int) -> None:
        """Record the cache payload that was restored into the fake backend."""
        self.loaded.append((cache_id, list(buffer)))

    def dump_cache_memory_to(self, cache_dir: str, cache_id: int) -> None:
        del cache_dir, cache_id

    def load_cache_memory_from(self, cache_dir: str, cache_id: int) -> None:
        del cache_dir, cache_id


def test_whisper_beam_debug_trace_predicate_follows_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whisper beam debug trace predicate should only reflect the trace-path environment variable."""
    monkeypatch.delenv("MBLT_WHISPER_BEAM_DEBUG_TRACE", raising=False)

    assert is_whisper_beam_debug_trace_enabled() is False

    monkeypatch.setenv("MBLT_WHISPER_BEAM_DEBUG_TRACE", "beam_trace.jsonl")

    assert is_whisper_beam_debug_trace_enabled() is True


def test_append_whisper_beam_debug_event_is_noop_without_trace_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Whisper beam debug event append should be a no-op when tracing is disabled."""
    monkeypatch.delenv("MBLT_WHISPER_BEAM_DEBUG_TRACE", raising=False)
    trace_path = tmp_path / "beam_trace.jsonl"

    append_whisper_beam_debug_event({"event": "noop"})

    assert not trace_path.exists()


def test_dump_cache_memory_roundtrip_restores_seq_length() -> None:
    """Restore in-memory cache snapshots with their original sequence length."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model)

    cache.set_seq_length(7)
    cache.dump_cache_memory()
    cache.set_seq_length(0)

    cache.load_cache_memory()

    assert cache.get_seq_length() == 7
    assert mxq_model.loaded == [(0, [b"cache-0"])]


def test_new_cache_for_reused_model_starts_with_fresh_logical_cursors() -> None:
    """A new request must not inherit the prior request's addressable KV prefix."""
    mxq_model = _FakeMxqModel()
    previous_request = MobilintCache(mxq_model, batch_size=2)
    previous_request.set_seq_length({0: 7, 1: 11})

    next_request = MobilintCache(mxq_model, batch_size=2)

    assert [next_request.get_seq_length(index=i) for i in range(2)] == [0, 0]


def test_fake_prefill_sets_seq_length_without_cache_buffer() -> None:
    """Fake prefill should only expose sequence length and clear cache payloads."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model)

    cache.set_seq_length(7)
    cache.dump_cache_memory()

    cache.fake_prefill(128)

    assert cache.get_seq_length() == 128
    assert cache.layers[0].buffer == []
    assert cache.layers[0].buffer_seq_length is None
    assert mxq_model.loaded == []


def test_fake_prefill_sets_batched_seq_lengths() -> None:
    """Fake prefill should support per-cache-id lengths for batched decode."""
    cache = MobilintCache(_FakeMxqModel(), batch_size=2)

    cache.fake_prefill({0: 32, 1: 64})

    assert cache.get_seq_length(index=0) == 32
    assert cache.get_seq_length(index=1) == 64


def test_fake_prefill_scalar_sets_all_batched_seq_lengths() -> None:
    """Scalar fake prefill should prepare every cache entry for batched decode."""
    cache = MobilintCache(_FakeMxqModel(), batch_size=3)

    cache.fake_prefill(128)

    assert [cache.get_seq_length(index=i) for i in range(3)] == [128, 128, 128]


def test_fake_prefill_rejects_negative_length() -> None:
    """Fake prefill should reuse sequence length validation."""
    cache = MobilintCache(_FakeMxqModel())

    with pytest.raises(ValueError, match="non-negative"):
        cache.fake_prefill(-1)


def test_beam_cache_reorder_works_with_token_histories() -> None:
    """Generic beam cache should reorder token histories for beam search."""
    cache = MobilintBeamCache(_FakeMxqModel(), batch_size=2)
    cache.commit_beam_tokens(0, [10, 11])
    cache.commit_beam_tokens(1, [20, 21, 22])

    cache.reorder_cache(torch.tensor([1, 0], dtype=torch.long))

    assert [cache.get_beam_tokens(i) for i in range(2)] == [[20, 21, 22], [10, 11]]
    assert [cache.get_seq_length(index=i) for i in range(2)] == [3, 2]


def test_whisper_cache_is_beam_cache_for_backwards_compatibility() -> None:
    """Whisper cache should preserve its public name while reusing generic beam cache behavior."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=1)

    assert isinstance(cache, MobilintBeamCache)


def test_whisper_cache_reorder_rejects_invalid_beam_idx_shape() -> None:
    """Whisper beam reorder should only accept rank-1 beam indices."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=2)

    with pytest.raises(ValueError, match="rank 1"):
        cache.reorder_cache(torch.tensor([[0, 1]], dtype=torch.long))


def test_whisper_cache_reorder_reorders_token_histories() -> None:
    """Whisper beam reorder should reorder token histories."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=3)
    cache.commit_beam_tokens(0, [10, 11, 12, 13])
    cache.commit_beam_tokens(1, [20, 21, 22, 23, 24])
    cache.commit_beam_tokens(2, [30, 31, 32, 33, 34, 35])

    result = cache.reorder_cache(torch.tensor([2, 0, 2], dtype=torch.long))

    assert result is cache
    assert [cache.get_beam_tokens(i) for i in range(3)] == [
        [30, 31, 32, 33, 34, 35],
        [10, 11, 12, 13],
        [30, 31, 32, 33, 34, 35],
    ]
    cache._beam_token_histories[0][0] = 99
    assert cache.get_beam_tokens(2) == [30, 31, 32, 33, 34, 35]
    assert [cache.get_seq_length(index=i) for i in range(3)] == [6, 4, 6]


def test_whisper_cache_reorder_identity_order_is_noop() -> None:
    """Whisper beam reorder should no-op for identity beam order."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=1)
    cache.commit_beam_tokens(0, [10, 11])
    cache.commit_beam_tokens(1, [20, 21])
    cache.commit_beam_tokens(2, [30, 31])

    result = cache.reorder_cache(torch.tensor([0, 1, 2], dtype=torch.long))

    assert result is cache
    assert cache.batch_size == 3
    assert [cache.get_beam_tokens(i) for i in range(3)] == [[10, 11], [20, 21], [30, 31]]
    assert [cache.get_seq_length(index=i) for i in range(3)] == [2, 2, 2]


def test_whisper_cache_copy_preserves_token_histories_safely() -> None:
    """Whisper cache copy should clone token histories without sharing mutable lists."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=2)
    cache.commit_beam_tokens(0, [10, 11])
    cache.commit_beam_tokens(1, [20, 21, 22])

    copied = cache.copy()
    cache._beam_token_histories[0][0] = 99
    cache._beam_seq_lengths[0] = 99

    assert isinstance(copied, MobilintWhisperCache)
    assert [copied.get_beam_tokens(i) for i in range(2)] == [[10, 11], [20, 21, 22]]
    assert copied._beam_seq_lengths == [2, 3]


def test_whisper_cache_tracks_encoder_source_count() -> None:
    """Whisper cache should preserve original encoder source count across copies only."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=2)

    cache.set_encoder_source_count(2)
    copied = cache.copy()
    cache.reset()

    assert copied.get_encoder_source_count() == 2
    assert cache.get_encoder_source_count() is None


def test_whisper_cache_rejects_invalid_encoder_source_count() -> None:
    """Whisper cache should require positive encoder source counts."""
    cache = MobilintWhisperCache(_FakeMxqModel(), batch_size=1)

    with pytest.raises(ValueError, match="positive"):
        cache.set_encoder_source_count(0)


def test_deepstack_cache_returns_real_chunk() -> None:
    """Deepstack cache should slice the current forward-call tensor."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=2, hidden_size=3)
    deepstack = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    cache.set_deepstack_tensor(deepstack)

    chunk = cache.get_deepstack_chunk(1, 3, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(chunk, deepstack[:, 1:3, :])


def test_deepstack_cache_fake_prefill_returns_zero_chunk() -> None:
    """Fake-prefilled deepstack cache should lazily provide zero decode chunks."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=2, hidden_size=3)

    cache.fake_prefill(128)
    chunk = cache.get_deepstack_chunk(0, 1, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.get_seq_length() == 128
    assert chunk.shape == (2, 1, 3)
    assert torch.count_nonzero(chunk).item() == 0


def test_deepstack_cache_reset_clears_deepstack_tensor() -> None:
    """Reset should clear per-call deepstack payloads and sequence length."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=1, hidden_size=2)
    cache.set_deepstack_tensor(torch.ones(1, 2, 2))
    cache.set_seq_length(4)

    cache.reset()
    chunk = cache.get_deepstack_chunk(0, 2, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.get_seq_length() == 0
    assert torch.count_nonzero(chunk).item() == 0


def test_deepstack_cache_multi_model_routes_rows_across_slots() -> None:
    """Multi-slot deepstack cache should route flat rows to their owning ``qbruntime.Model``."""
    models = [_FakeMxqModel() for _ in range(2)]
    cache = MobilintDeepStackCache(
        models,
        per_model_batch=1,
        num_deepstack_layers=1,
        hidden_size=2,
    )

    assert cache.n_models == 2
    assert cache.k_per_model == 1
    assert cache.batch_size == 2
    assert cache.slot_of(0) == (0, 0)
    assert cache.slot_of(1) == (1, 0)
    assert cache.model_of(0) is models[0]
    assert cache.model_of(1) is models[1]
    # The deepstack payload plumbing is preserved end-to-end on a multi-slot cache.
    cache.set_deepstack_tensor(torch.ones(1, 4, 2))
    chunk = cache.get_deepstack_chunk(0, 2, device=torch.device("cpu"), dtype=torch.float32)
    assert chunk.shape == (1, 2, 2)


def test_build_mobilint_cache_from_model_dispatches_deepstack_cache_to_slots() -> None:
    """``build_mobilint_cache_from_model`` must forward ``MobilintDeepStackCache`` extras to slots.

    Regression for PR #109 Codex P1: ``MobilintQwen3VLTextModel._get_cache`` used to build the
    deepstack cache from ``get_cache_mxq_model()`` (slot 0), so a multi-slot text backend never
    received rows ``1..B-1``. The shared factory now handles ``cache_cls=MobilintDeepStackCache``
    plus its keyword-only ``num_deepstack_layers`` / ``hidden_size`` extras.
    """

    class _StubBackend:
        def __init__(self, models: list) -> None:
            self.mxq_models = models
            self.k_per_model = 1

    class _StubModel:
        def __init__(self, models: list) -> None:
            self.npu_backend = _StubBackend(models)

    models = [_FakeMxqModel() for _ in range(3)]
    stub_model = _StubModel(models)
    cache = build_mobilint_cache_from_model(
        stub_model,
        batch_size=3,
        cache_cls=MobilintDeepStackCache,
        num_deepstack_layers=2,
        hidden_size=4,
    )
    assert isinstance(cache, MobilintDeepStackCache)
    assert cache.n_models == 3
    assert cache.k_per_model == 1
    assert cache.batch_size == 3
    assert cache.num_deepstack_layers == 2
    assert cache.hidden_size == 4
    for row in range(3):
        assert cache.model_of(row) is models[row]


def test_eagle3_cache_tracks_base_and_draft_lengths_independently() -> None:
    """EAGLE-3 cache should track base and draft MXQ sequence lengths separately."""
    cache = MobilintEagle3Cache(_FakeMxqModel(), _FakeMxqModel())

    cache.set_base_seq_length(12)
    cache.set_draft_seq_length(7)

    assert cache.get_base_seq_length() == 12
    assert cache.get_draft_seq_length() == 7


def test_eagle3_cache_reset_clears_tree_state() -> None:
    """EAGLE-3 cache reset should clear speculative decoding state."""
    cache = MobilintEagle3Cache(_FakeMxqModel(), _FakeMxqModel())
    cache.accept_tokens = torch.ones(1, 2, dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 2, 2)
    cache.retrieve_indices = torch.ones(1, 2, dtype=torch.long)
    cache.tree_position_ids = torch.ones(2, dtype=torch.long)
    cache.pending_draft_tokens = torch.ones(1, 2, dtype=torch.long)

    cache.reset()

    assert cache.accept_tokens is None
    assert cache.tree_mask is None
    assert cache.retrieve_indices is None
    assert cache.tree_position_ids is None
    assert cache.pending_draft_tokens is None


def test_eagle3_cache_copy_clears_tree_state_but_preserves_seq_lengths() -> None:
    """EAGLE-3 cache copy should drop transient tree state and keep committed lengths."""
    cache = MobilintEagle3Cache(_FakeMxqModel(), _FakeMxqModel())
    cache.set_base_seq_length(4)
    cache.set_draft_seq_length(3)
    cache.accept_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 2, 2)

    copied = cache.copy()

    assert copied.get_base_seq_length() == 4
    assert copied.get_draft_seq_length() == 3
    assert copied.accept_tokens is None
    assert copied.tree_mask is None
    assert copied.retrieve_indices is None
    assert copied.tree_position_ids is None
    assert copied.pending_draft_tokens is None


def test_eagle3_cache_dump_load_roundtrip_restores_base_and_draft_seq_lengths() -> None:
    """EAGLE-3 cache dump/load round-trip should restore both cache layers."""
    base_mxq = _FakeMxqModel()
    draft_mxq = _FakeMxqModel()
    cache = MobilintEagle3Cache(base_mxq, draft_mxq)

    cache.set_base_seq_length(11)
    cache.set_draft_seq_length(7)
    cache.dump_cache_memory()
    cache.set_base_seq_length(0)
    cache.set_draft_seq_length(0)

    cache.load_cache_memory()

    assert cache.get_base_seq_length() == 11
    assert cache.get_draft_seq_length() == 7
    assert base_mxq.loaded == [(0, [b"cache-0"])]
    assert draft_mxq.loaded == [(0, [b"cache-0"])]


def test_mobilint_cache_legacy_batch_size_promotes_single_model_to_n1_k8() -> None:
    """Legacy ``batch_size=8`` on a single Model should build one N=1, K=8 cache."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model, batch_size=8)

    assert cache.n_models == 1
    assert cache.k_per_model == 8
    assert cache.batch_size == 8
    assert len(cache.layers) == 8
    assert all(layer.mxq_model is mxq_model for layer in cache.layers)
    assert cache.mxq_model is mxq_model
    assert cache.slot_of(0) == (0, 0)
    assert cache.slot_of(7) == (0, 7)


def test_mobilint_cache_two_models_per_model_batch_one_yields_flat_row_layout() -> None:
    """Two Models with ``per_model_batch=1`` should map row i to model i, slot 0."""
    model_0 = _FakeMxqModel()
    model_1 = _FakeMxqModel()
    cache = MobilintCache([model_0, model_1], per_model_batch=1)

    assert cache.n_models == 2
    assert cache.k_per_model == 1
    assert cache.batch_size == 2
    assert cache.slot_of(0) == (0, 0)
    assert cache.slot_of(1) == (1, 0)
    assert cache.model_of(0) is model_0
    assert cache.model_of(1) is model_1
    assert cache.layers[0].mxq_model is model_0
    assert cache.layers[0].cache_id == 0
    assert cache.layers[1].mxq_model is model_1
    assert cache.layers[1].cache_id == 0


def test_mobilint_cache_four_models_per_model_batch_sixteen_slot_math() -> None:
    """Four Models × K=16 should produce 64 flat rows with the divmod slot layout."""
    models = [_FakeMxqModel() for _ in range(4)]
    cache = MobilintCache(models, per_model_batch=16)

    assert cache.n_models == 4
    assert cache.k_per_model == 16
    assert cache.batch_size == 64
    assert len(cache.layers) == 64
    assert cache.slot_of(17) == (1, 1)
    assert cache.model_of(17) is models[1]
    assert cache.slot_of(63) == (3, 15)
    assert cache.model_of(63) is models[3]


def test_mobilint_cache_group_by_model_preserves_row_order_per_model() -> None:
    """group_by_model should bucket flat rows by owning Model in insertion order."""
    models = [_FakeMxqModel() for _ in range(3)]
    cache = MobilintCache(models, per_model_batch=4)

    grouped = cache.group_by_model([9, 0, 5, 2, 4])

    assert grouped == {
        2: [(9, 1)],
        0: [(0, 0), (2, 2)],
        1: [(5, 1), (4, 0)],
    }


def test_mobilint_cache_update_seen_tokens_per_row_routes_via_slot_of() -> None:
    """update_seen_tokens with a dict should route to the correct (model, cache_id) layer."""
    models = [_FakeMxqModel() for _ in range(2)]
    cache = MobilintCache(models, per_model_batch=1)

    cache.update_seen_tokens({0: 5, 1: 3})

    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 3


def test_mobilint_cache_ensure_batch_size_rejects_multi_model_growth() -> None:
    """ensure_batch_size beyond N*K must fail for multi-Model caches."""
    cache = MobilintCache([_FakeMxqModel(), _FakeMxqModel()], per_model_batch=1)

    with pytest.raises(ValueError, match="multi-Model"):
        cache.ensure_batch_size(4)


def test_mobilint_cache_ensure_batch_size_grows_single_model_hardware_batch() -> None:
    """ensure_batch_size on an N=1 cache should keep the legacy hardware-batch growth."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model, per_model_batch=2)

    cache.ensure_batch_size(5)

    assert cache.n_models == 1
    assert cache.batch_size == 5
    assert cache.k_per_model == 5
    assert [layer.cache_id for layer in cache.layers] == [0, 1, 2, 3, 4]
    assert all(layer.mxq_model is mxq_model for layer in cache.layers)


def test_mobilint_cache_rejects_conflicting_batch_size_and_per_model_batch() -> None:
    """Passing both per_model_batch and legacy batch_size should raise."""
    with pytest.raises(TypeError, match="not both"):
        MobilintCache(_FakeMxqModel(), per_model_batch=2, batch_size=3)


def test_mobilint_cache_rejects_legacy_batch_size_with_multi_model_list() -> None:
    """Multi-Model list + legacy batch_size must raise; it silently misroutes slots.

    ``MobilintCache([m0, m1], batch_size=B)`` was previously accepted as ``K = B``,
    producing ``2 * B`` rows where the first ``B`` rows landed entirely on model 0
    and the second ``B`` rows entirely on model 1 — defeating the caller's intended
    "batch of B distributed across 2 slots". Reject at construction with a clear
    pointer at ``per_model_batch``.
    """
    with pytest.raises(TypeError, match="per_model_batch"):
        MobilintCache([_FakeMxqModel(), _FakeMxqModel()], batch_size=4)


def test_mobilint_cache_copy_preserves_multi_model_layout_and_seq_lengths() -> None:
    """copy() should keep the same Model list identity and layer sequence lengths."""
    models = [_FakeMxqModel() for _ in range(2)]
    cache = MobilintCache(models, per_model_batch=3)
    cache.set_seq_length({0: 4, 5: 7})

    copied = cache.copy()

    assert copied.n_models == 2
    assert copied.k_per_model == 3
    assert copied.batch_size == 6
    assert copied.mxq_models[0] is models[0]
    assert copied.mxq_models[1] is models[1]
    assert copied.get_seq_length(0) == 4
    assert copied.get_seq_length(5) == 7
    assert copied.slot_of(5) == (1, 2)


def test_mobilint_beam_cache_rejects_multi_model_dispatch() -> None:
    """Beam cache should refuse N > 1 because encoder-decoder tracking is N=1 only."""
    with pytest.raises(NotImplementedError, match="multi-Model"):
        MobilintBeamCache([_FakeMxqModel(), _FakeMxqModel()])


def test_mobilint_beam_cache_accepts_single_element_list() -> None:
    """Beam cache should accept a length-1 list because it stays N=1."""
    cache = MobilintBeamCache([_FakeMxqModel()], batch_size=2)

    assert cache.n_models == 1
    assert cache.batch_size == 2
