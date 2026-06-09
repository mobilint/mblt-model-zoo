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
