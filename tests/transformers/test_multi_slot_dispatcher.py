"""Direct unit tests for :class:`MultiSlotDispatcher`.

Exercises the dispatcher against a mock backend rather than the full
``MobilintModelMixin`` mixin stack, so the routing + merge contract can be
asserted independently of the caller's chunk-assembly logic.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher


class _StaticN1Mxq:
    """Static-token-axis MXQ stub emitting per-item last-row logits."""

    def __init__(self, vocab_size: int = 5, tag: float = 0.0) -> None:
        self.vocab_size = vocab_size
        self.tag = float(tag)
        self.calls: list[dict] = []

    def get_model_output_shape(self):
        return [(1, 1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params):
        chunk = np.asarray(inputs[0])
        n_items = len(batch_params)
        payload = np.stack(
            [
                np.full(
                    (self.vocab_size,),
                    self.tag + 10.0 * p.cache_id + 0.1 * local_row,
                    dtype=np.float32,
                )
                for local_row, p in enumerate(batch_params)
            ],
            axis=0,
        )
        assert payload.shape == (n_items, self.vocab_size)
        self.calls.append(
            {
                "shape": tuple(chunk.shape),
                "batch": [(p.cache_id, int(p.sequence_length), int(p.cache_size)) for p in batch_params],
            }
        )
        return [payload]


class _DynamicMxq:
    """Dynamic-token-axis MXQ stub emitting per-token flat logits."""

    def __init__(self, vocab_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def get_model_output_shape(self):
        return [(1, -1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params):
        chunk = np.asarray(inputs[0])
        total_tokens = sum(int(p.sequence_length) for p in batch_params)
        rows: list[np.ndarray] = []
        for p in batch_params:
            for token_idx in range(int(p.sequence_length)):
                rows.append(
                    np.full(
                        (self.vocab_size,),
                        p.cache_id * 1000.0 + token_idx,
                        dtype=np.float32,
                    )
                )
        flat = np.stack(rows, axis=0)
        assert flat.shape == (total_tokens, self.vocab_size)
        self.calls.append(
            {
                "shape": tuple(chunk.shape),
                "batch": [(p.cache_id, int(p.sequence_length), int(p.cache_size)) for p in batch_params],
            }
        )
        return [flat]


class _MockBackend:
    """Minimal backend duck: ``mxq_models``, ``k_per_model``, ``output_layout``."""

    def __init__(self, mxq_models: List[object], k_per_model: int = 1) -> None:
        self.mxq_models = list(mxq_models)
        self.k_per_model = int(k_per_model)
        self._output_layout_cached: Optional[Literal["n_items", "n_tokens"]] = None

    @property
    def output_layout(self) -> Optional[Literal["n_items", "n_tokens"]]:
        cached = self._output_layout_cached
        if cached is not None:
            return cached
        shapes = self.mxq_models[0].get_model_output_shape()
        if not shapes:
            return None
        first_shape = tuple(shapes[0])
        if len(first_shape) < 2:
            return None
        token_axis = int(first_shape[-2])
        return "n_tokens" if token_axis == -1 else "n_items"

    def _set_output_layout(self, layout: Literal["n_items", "n_tokens"]) -> None:
        self._output_layout_cached = layout


def _make_embed(row_count: int, hidden: int = 4) -> torch.Tensor:
    return torch.arange(row_count * hidden, dtype=torch.float32).reshape(row_count, hidden)


# ---------------------------------------------------------------------------
# slot_of routing
# ---------------------------------------------------------------------------


def test_slot_of_uses_past_key_values_when_available() -> None:
    """MobilintCache.slot_of is the source of truth for row routing."""
    m0, m1 = _StaticN1Mxq(), _StaticN1Mxq()
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _StubCache:
        def slot_of(self, row: int) -> Tuple[int, int]:
            # Reverse routing so we can spot the fall-through to divmod.
            if row == 0:
                return (1, 0)
            if row == 1:
                return (0, 0)
            raise IndexError(row)

    assert dispatcher.slot_of(0, past_key_values=_StubCache()) == (1, 0)
    assert dispatcher.slot_of(1, past_key_values=_StubCache()) == (0, 0)


def test_slot_of_falls_back_to_divmod_when_cache_missing() -> None:
    """Without a cache the dispatcher must derive the mapping from ``k_per_model``."""
    m0, m1 = _StaticN1Mxq(), _StaticN1Mxq()
    backend = _MockBackend([m0, m1], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    # k=2 flattens rows 0..1 -> slot 0, rows 2..3 -> slot 1.
    assert dispatcher.slot_of(0) == (0, 0)
    assert dispatcher.slot_of(1) == (0, 1)
    assert dispatcher.slot_of(2) == (1, 0)
    assert dispatcher.slot_of(3) == (1, 1)


# ---------------------------------------------------------------------------
# Single-group fast path
# ---------------------------------------------------------------------------


def test_dispatch_single_group_fast_path_uses_one_infer_call() -> None:
    """A backend with N==1 must skip the ThreadPoolExecutor and issue one blocking infer."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=7.0)
    backend = _MockBackend([m0], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    merged, shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[4, 4],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
    )

    assert merged.shape == (2, 3)
    # Every row picks up ``tag + 10 * cache_id + 0.1 * local_row``.
    assert m0.calls[0]["batch"] == [(0, 1, 4), (1, 1, 4)]
    assert shape[-1] == 4  # hidden axis pass-through
    # Two rows, one vocab-3 output apiece.
    assert merged[0][0] == pytest.approx(7.0)
    assert merged[1][0] == pytest.approx(17.1)


# ---------------------------------------------------------------------------
# Multi-group parallel dispatch
# ---------------------------------------------------------------------------


def test_dispatch_multi_group_preserves_caller_row_order() -> None:
    """Rows routed to different Model slots must still merge back in caller order."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _RoutingCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
        past_key_values=_RoutingCache(),
    )

    assert merged.shape == (2, 3)
    # Row 0 came from m0 (tag=0) and row 1 came from m1 (tag=100), untouched by
    # the merge even though threads may have finished out of order.
    assert merged[0][0] == pytest.approx(0.0)
    assert merged[1][0] == pytest.approx(100.0)
    assert len(m0.calls) == 1
    assert len(m1.calls) == 1


def test_dispatch_multi_group_with_four_groups() -> None:
    """Merge routing must scale across four Model slots dispatched in parallel."""
    ms = [_StaticN1Mxq(vocab_size=3, tag=1000.0 * i) for i in range(4)]
    backend = _MockBackend(ms, k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _RoutingCache:
        n_models = 4
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1, 2, 3],
        sequence_lengths=[1, 1, 1, 1],
        cache_sizes=[0, 0, 0, 0],
        inputs_embeds_chunks=[_make_embed(1) for _ in range(4)],
        max_sequence_length=1,
        past_key_values=_RoutingCache(),
    )

    assert merged.shape == (4, 3)
    for row_idx, mxq in enumerate(ms):
        # ``tag + 10 * cache_id`` where local cache_id is 0 for every slot here.
        assert merged[row_idx][0] == pytest.approx(1000.0 * row_idx)
        assert len(mxq.calls) == 1


# ---------------------------------------------------------------------------
# Layout selection
# ---------------------------------------------------------------------------


def test_dispatch_n_tokens_layout_preserves_per_token_rows_across_groups() -> None:
    """Dynamic-axis MXQs emit per-token flat rows; merge must respect caller offsets.

    Reproduces the P1 regression: first group is a size-1 row (ambiguous), second
    group has three tokens. Merge must not truncate the longer group.
    """
    m0 = _DynamicMxq(vocab_size=4)
    m1 = _DynamicMxq(vocab_size=4)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _RoutingCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 3],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(3)],
        max_sequence_length=3,
        past_key_values=_RoutingCache(),
    )

    # Total tokens = 4 (1 + 3); vocab = 4.
    assert merged.shape == (4, 4)
    # Row 0's single token from m0 (cache_id=0 -> tag 0.0).
    assert merged[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    # Row 1's three tokens from m1 (cache_id=0 within m1 -> 0*1000 + token_idx).
    assert merged[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert merged[2].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert merged[3].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_dispatch_runtime_layout_fallback_pins_backend_cache() -> None:
    """When the compile-time probe returns ``None`` the dispatcher must pin the answer.

    The mock backend's ``output_layout`` returns ``None`` on first read; after
    one dispatch the layout is resolved from an unambiguous group and cached
    on the backend so subsequent dispatches skip re-inspection.
    """
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)

    class _NoProbeBackend(_MockBackend):
        @property
        def output_layout(self):
            return self._output_layout_cached

    backend = _NoProbeBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _RoutingCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    assert backend.output_layout is None

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[3, 3],  # unambiguous: n_items=2 != n_tokens=3 per group
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(3), _make_embed(3)],
        max_sequence_length=3,
        past_key_values=_RoutingCache(),
    )

    assert merged.shape == (2, 3)
    # Backend layout was pinned to ``n_items`` by the runtime fallback.
    assert backend.output_layout == "n_items"


def test_dispatch_reoverrides_stale_n_tokens_cache_from_runtime_observation() -> None:
    """A ``K > 1`` batched MXQ whose probe misfired must be re-pinned at runtime.

    Regression (P1): ``MobilintNPUBackend._probe_output_layout`` reads the
    token axis at position ``-2`` of the compiled shape. For a batched MXQ
    (K=16) whose batch axis is reported ``-1`` at that position, the probe
    incorrectly pinned ``"n_tokens"``. ``_merge_group_outputs`` then sliced
    per-item last-row outputs as if they were per-token flat, producing a
    broadcast error on multi-slot dispatch.

    This test simulates the failure: prime the backend cache with the wrong
    ``"n_tokens"`` layout, dispatch a multi-slot prefill batch whose raw
    outputs are per-item (``_StaticN1Mxq`` shape), and assert the dispatcher
    overwrites the cache with the observed ``"n_items"`` layout and merges
    without error.
    """
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    # Simulate the probe misfire: cache "n_tokens" even though the fake MXQs
    # emit per-item rows.
    backend._set_output_layout("n_tokens")
    assert backend.output_layout == "n_tokens"

    class _RoutingCache:
        # k=2 splits rows 0/1 -> slot 0, rows 2/3 -> slot 1. Multi-token
        # sequence lengths guarantee unambiguous observation.
        n_models = 2
        k_per_model = 2

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 2)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1, 2, 3],
        sequence_lengths=[3, 3, 3, 3],
        cache_sizes=[0, 0, 0, 0],
        inputs_embeds_chunks=[_make_embed(3) for _ in range(4)],
        max_sequence_length=3,
        past_key_values=_RoutingCache(),
    )

    # Merge succeeded with the corrected layout, one row per caller item.
    assert merged.shape == (4, 3)
    # Cache was overwritten to the observed layout.
    assert backend.output_layout == "n_items"
    # Row order preserved. ``_StaticN1Mxq`` fills every vocab entry with
    # ``tag + 10 * local_cache_id + 0.1 * local_row_within_group``, so:
    #   row 0 -> m0, local_cache_id=0, local_row=0 -> 0.0
    #   row 1 -> m0, local_cache_id=1, local_row=1 -> 10.1
    #   row 2 -> m1, local_cache_id=0, local_row=0 -> 100.0
    #   row 3 -> m1, local_cache_id=1, local_row=1 -> 110.1
    assert merged[0][0] == pytest.approx(0.0)
    assert merged[1][0] == pytest.approx(10.1)
    assert merged[2][0] == pytest.approx(100.0)
    assert merged[3][0] == pytest.approx(110.1)


# ---------------------------------------------------------------------------
# Cacheless capacity guard
# ---------------------------------------------------------------------------


def test_dispatch_cacheless_within_capacity_n1_k1_single_row() -> None:
    """Regression: n_slots=1, k_per_model=1, cache_ids=[0] must dispatch cleanly."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    backend = _MockBackend([m0], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0],
        sequence_lengths=[1],
        cache_sizes=[0],
        inputs_embeds_chunks=[_make_embed(1)],
        max_sequence_length=1,
    )
    assert merged.shape == (1, 3)


def test_dispatch_cacheless_over_capacity_n1_k1_raises_value_error() -> None:
    """n_slots=1, k_per_model=1, cache_ids=[0, 1]: two items exceeds N*K=1."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    backend = _MockBackend([m0], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1],
            sequence_lengths=[1, 1],
            cache_sizes=[0, 0],
            inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
            max_sequence_length=1,
        )

    msg = str(excinfo.value)
    assert "Cacheless batched dispatch exceeds backend capacity" in msg
    assert "n_items=2" in msg
    assert "N*K = 1 * 1 = 1" in msg


def test_dispatch_cacheless_within_capacity_n2_k1_regression() -> None:
    """n_slots=2, k_per_model=1, cache_ids=[0, 1] matches capacity — must succeed."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
    )
    assert merged.shape == (2, 3)


def test_dispatch_cacheless_max_row_beyond_capacity_raises_value_error() -> None:
    """n_slots=2, k_per_model=1, cache_ids=[2]: single item but row_id=2 exceeds N*K=2."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[2],
            sequence_lengths=[1],
            cache_sizes=[0],
            inputs_embeds_chunks=[_make_embed(1)],
            max_sequence_length=1,
        )

    msg = str(excinfo.value)
    assert "max row_id=2" in msg
    assert "N*K = 2 * 1 = 2" in msg


def test_dispatch_cacheless_within_capacity_n1_k4_regression() -> None:
    """n_slots=1, k_per_model=4, cache_ids=[0,1,2,3] fills exactly one batched slot."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    backend = _MockBackend([m0], k_per_model=4)
    dispatcher = MultiSlotDispatcher(backend)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1, 2, 3],
        sequence_lengths=[1, 1, 1, 1],
        cache_sizes=[0, 0, 0, 0],
        inputs_embeds_chunks=[_make_embed(1) for _ in range(4)],
        max_sequence_length=1,
    )
    assert merged.shape == (4, 3)


def test_dispatch_with_cache_over_backend_capacity_delegates_to_cache() -> None:
    """Regression: with a supplied cache the capacity guard is inactive.

    The dispatcher must delegate slot resolution to ``past_key_values.slot_of``
    and not raise the cacheless-only ``ValueError`` even when caller row IDs
    exceed the backend's ``N * K``. Validating the with-cache case is
    ``_validate_batch_cache``'s job, invoked earlier by the caller.
    """
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _RoutingCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            # Route rows 0 -> m0, 1 -> m1 (matches N*K=2).
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
        past_key_values=_RoutingCache(),
    )

    assert merged.shape == (2, 3)


# ---------------------------------------------------------------------------
# N==1 guard
# ---------------------------------------------------------------------------


def test_assert_single_slot_raises_on_multi_slot_backend() -> None:
    m0 = _StaticN1Mxq()
    m1 = _StaticN1Mxq()
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    with pytest.raises(NotImplementedError) as excinfo:
        dispatcher.assert_single_slot("TestCaller", "Remediation text.")

    msg = str(excinfo.value)
    assert "TestCaller" in msg
    assert "N=2" in msg
    assert "Remediation text." in msg


def test_assert_single_slot_passes_on_n1_backend() -> None:
    backend = _MockBackend([_StaticN1Mxq()], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)
    dispatcher.assert_single_slot("TestCaller", "n/a")


# ---------------------------------------------------------------------------
# pack_extra_inputs hook receives flat rows
# ---------------------------------------------------------------------------


def test_pack_extra_inputs_receives_flat_row_cache_ids() -> None:
    """The extras hook must see caller-visible flat rows, not local slot ids."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    seen_ids: list[list[int]] = []

    def _extras(*, chunk_start, sequence_lengths_chunks, cache_ids):
        seen_ids.append(list(cache_ids))
        # Return a dummy extra shaped ``(1, total_tokens, 1)`` so the fake
        # ignoring backend keeps working.
        total = sum(sequence_lengths_chunks)
        return [np.zeros((1, total, 1), dtype=np.float32)]

    class _RoutingCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
        past_key_values=_RoutingCache(),
        pack_extra_inputs=_extras,
    )

    # Both groups saw their respective flat row (0 or 1), NOT local id 0/0.
    assert sorted(seen_ids) == [[0], [1]]


# ---------------------------------------------------------------------------
# Topology + Model-handle identity validation
# ---------------------------------------------------------------------------


def test_dispatch_rejects_cache_with_n_models_mismatch_but_matching_aggregate() -> None:
    """Cache ``(N=1, K=2)`` and backend ``(N=2, K=1)`` share aggregate 2 but route incompatibly."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _MismatchedCache:
        # Cache built as (N=1, K=2): capacity 2 same as backend, but a
        # cache.slot_of(1) would return (0, 1) -> route both rows to m0.
        n_models = 1
        k_per_model = 2
        mxq_models = [m0]

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 2)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1],
            sequence_lengths=[1, 1],
            cache_sizes=[0, 0],
            inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
            max_sequence_length=1,
            past_key_values=_MismatchedCache(),
        )

    msg = str(excinfo.value)
    assert "n_models" in msg
    assert "cache.n_models=1" in msg
    assert "backend n_slots=2" in msg


def test_dispatch_rejects_cache_with_k_per_model_mismatch_but_matching_aggregate() -> None:
    """Cache ``(N=2, K=1)`` and backend ``(N=1, K=2)`` share aggregate 2 (symmetric mismatch)."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    backend = _MockBackend([m0], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    other_m = _StaticN1Mxq(vocab_size=3, tag=100.0)

    class _MismatchedCache:
        n_models = 2
        k_per_model = 1
        mxq_models = [m0, other_m]

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1],
            sequence_lengths=[1, 1],
            cache_sizes=[0, 0],
            inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
            max_sequence_length=1,
            past_key_values=_MismatchedCache(),
        )

    msg = str(excinfo.value)
    # ``n_models`` is checked first, so a same-aggregate (N=2,K=1) vs (N=1,K=2)
    # mismatch surfaces on the N axis. The K-axis message wording is still
    # verified below by the pure-K test where N matches.
    assert "n_models" in msg or "k_per_model" in msg


def test_dispatch_rejects_cache_with_k_axis_only_mismatch() -> None:
    """N agrees but cache.k_per_model != backend.k_per_model must raise on the K axis."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    class _MismatchedCache:
        n_models = 2
        k_per_model = 1
        mxq_models = [m0, m1]

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1],
            sequence_lengths=[1, 1],
            cache_sizes=[0, 0],
            inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
            max_sequence_length=1,
            past_key_values=_MismatchedCache(),
        )

    msg = str(excinfo.value)
    assert "k_per_model" in msg
    assert "cache.k_per_model=1" in msg
    assert "backend k_per_model=2" in msg


def test_dispatch_rejects_cache_with_stale_model_handles() -> None:
    """A cache carrying handles from a disposed prior backend must be rejected on identity."""
    # Live backend the dispatcher was built against.
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=2)
    dispatcher = MultiSlotDispatcher(backend)

    # Stale handles from a prior (disposed) backend: same shape, different objects.
    stale_m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    stale_m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)

    class _StaleCache:
        n_models = 2
        k_per_model = 2
        mxq_models = [stale_m0, stale_m1]

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 2)

    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1, 2, 3],
            sequence_lengths=[1, 1, 1, 1],
            cache_sizes=[0, 0, 0, 0],
            inputs_embeds_chunks=[_make_embed(1) for _ in range(4)],
            max_sequence_length=1,
            past_key_values=_StaleCache(),
        )

    msg = str(excinfo.value)
    assert "Model-handle identity mismatch" in msg


def test_dispatch_accepts_cache_without_mxq_models_when_topology_matches() -> None:
    """A stub cache exposing ``slot_of`` but no ``mxq_models`` must skip the identity check.

    Topology check still fires — the topology attributes are present and match
    — but the missing ``mxq_models`` attribute is treated as "identity unknown"
    and silently skipped so lightweight test doubles keep working.
    """
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _TopologyOnlyCache:
        n_models = 2
        k_per_model = 1

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
        past_key_values=_TopologyOnlyCache(),
    )
    assert merged.shape == (2, 3)


def test_dispatch_accepts_cache_with_matching_topology_and_handles() -> None:
    """Regression: matching ``(N, K)`` and identical Model handles must dispatch normally."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    class _MatchedCache:
        n_models = 2
        k_per_model = 1
        mxq_models = [m0, m1]

        def slot_of(self, row: int) -> Tuple[int, int]:
            return divmod(row, 1)

    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
        past_key_values=_MatchedCache(),
    )
    assert merged.shape == (2, 3)
    # Routing is unchanged: row 0 -> m0, row 1 -> m1.
    assert merged[0][0] == pytest.approx(0.0)
    assert merged[1][0] == pytest.approx(100.0)


def test_dispatch_cacheless_bypasses_topology_check() -> None:
    """``past_key_values=None`` must no-op the topology validator (cacheless guard still fires)."""
    m0 = _StaticN1Mxq(vocab_size=3, tag=0.0)
    m1 = _StaticN1Mxq(vocab_size=3, tag=100.0)
    backend = _MockBackend([m0, m1], k_per_model=1)
    dispatcher = MultiSlotDispatcher(backend)

    # Within capacity: cacheless N*K guard passes, topology check is skipped.
    merged, _shape = dispatcher.dispatch(
        cache_ids=[0, 1],
        sequence_lengths=[1, 1],
        cache_sizes=[0, 0],
        inputs_embeds_chunks=[_make_embed(1), _make_embed(1)],
        max_sequence_length=1,
    )
    assert merged.shape == (2, 3)

    # Over capacity: the sibling cacheless guard still fires unchanged.
    with pytest.raises(ValueError) as excinfo:
        dispatcher.dispatch(
            cache_ids=[0, 1, 2],
            sequence_lengths=[1, 1, 1],
            cache_sizes=[0, 0, 0],
            inputs_embeds_chunks=[_make_embed(1), _make_embed(1), _make_embed(1)],
            max_sequence_length=1,
        )
    assert "Cacheless batched dispatch exceeds backend capacity" in str(excinfo.value)
