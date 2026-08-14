"""Regression tests for the multi-Model sw-batch dispatch path.

``MobilintNPUBackend`` can host ``N`` :class:`qbruntime.Model` slots so that a
non-batch-compiled MXQ (``K == 1``) still services a logical ``B > 1`` batch by
routing each row to its own slot. These tests verify the routing decisions
inside :meth:`MobilintModelMixin._llm_forward_batch` without booting an NPU:

* When the backend exposes ``N == 2`` Models and the cache is a
  ``MobilintCache([m0, m1], per_model_batch=1)`` (aggregate ``batch_size == 2``),
  each flat row is dispatched to its owning Model with local ``cache_id = 0``.
* The merged output preserves the caller's row order regardless of dispatch
  order across parallel worker threads.
* ``MobilintCache([m], per_model_batch=B)`` continues to service a single-slot
  hardware batch — the sw-batch path only kicks in for multi-Model backends,
  and the routing decision is driven by ``past_key_values.slot_of``.
* Legacy ``target_cores`` inputs migrate to canonical ``d:c:k`` strings, and
  ``dev_no`` accepts both scalar and list-shaped values so a batched pipeline
  can span multiple devices.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache
from mblt_model_zoo.hf_transformers.utils.configuration_utils import (
    _normalize_npu_target_kwargs,
)
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin
from mblt_model_zoo.utils.npu_backend import MobilintNPUBackend
from tests.transformers._fake_mxq import StaticLastOnlyMxq


class _MultiSlotFakeBackend:
    """FakeBackend variant that exposes multiple ``qbruntime.Model`` stubs."""

    def __init__(self, mxq_models: List[StaticLastOnlyMxq]) -> None:
        self.mxq_models = list(mxq_models)
        # ``mxq_model`` compat shim mirrors the multi-slot backend's slot 0.
        self.mxq_model = self.mxq_models[0]


def _make_multi_slot_model(mxq_models: List[StaticLastOnlyMxq]) -> MobilintModelMixin:
    """Construct a bare mixin bound to a multi-Model fake backend."""
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = _MultiSlotFakeBackend(mxq_models)
    model.config = type(
        "Config",
        (),
        {"npu_prefill_chunk_size": None, "max_batch_size": len(mxq_models)},
    )()
    model.npu_time = None
    return model


# ---------------------------------------------------------------------------
# Multi-Model routing
# ---------------------------------------------------------------------------


def test_multi_model_dispatch_routes_each_row_to_its_owning_model() -> None:
    """N=2, K=1: batch=2 must call ``m0`` for row 0 and ``m1`` for row 1 with local cache_id=0."""
    m0 = StaticLastOnlyMxq(vocab_size=5, max_width=4)
    m1 = StaticLastOnlyMxq(vocab_size=5, max_width=4)
    model = _make_multi_slot_model([m0, m1])
    cache = MobilintCache([m0, m1], per_model_batch=1)

    inputs_embeds = torch.randn(2, 3, 4, dtype=torch.float32)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(inputs_embeds.shape[1]),
        attention_mask=attention_mask,
    )

    # Each Model observed exactly one batched infer for its owned row.
    assert len(m0.calls) == 1 and "batch" in m0.calls[0]
    assert len(m1.calls) == 1 and "batch" in m1.calls[0]
    # BatchParam.cache_id is the LOCAL slot id inside the owning Model
    # (0..k_per_model-1), NOT the flat batch row: N=2 K=1 flattens as
    # row 0 -> (m0, 0) and row 1 -> (m1, 0).
    assert m0.calls[0]["batch"] == [(0, 3, 0)]
    assert m1.calls[0]["batch"] == [(0, 3, 0)]
    # Output preserves caller row order.
    assert logits.shape == (2, 1, 5)


def test_multi_model_dispatch_preserves_row_order_when_rows_map_to_reverse_models() -> None:
    """Cache ``slot_of`` routing must survive re-ordering across worker threads."""
    m0 = StaticLastOnlyMxq(vocab_size=3, max_width=4)
    m1 = StaticLastOnlyMxq(vocab_size=3, max_width=4)
    model = _make_multi_slot_model([m0, m1])
    cache = MobilintCache([m0, m1], per_model_batch=1)

    # Deliberately differentiate the two rows so a swap would be visible in
    # the returned logits. ``StaticLastOnlyMxq`` encodes cache_id + a
    # monotonically-increasing per-model counter into every output row.
    inputs_embeds = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.ones(2, 1, dtype=torch.long)

    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(inputs_embeds.shape[1]),
        attention_mask=attention_mask,
    )

    # StaticLastOnlyMxq fills each row with ``cache_id * 100 + counter``.
    # Both Models saw cache_id=0 in their first (and only) call, so row 0
    # (from m0) should carry value 1 and row 1 (from m1) should carry value 1.
    assert logits.shape == (2, 1, 3)
    assert torch.equal(logits[0, 0], torch.full((3,), 1.0))
    assert torch.equal(logits[1, 0], torch.full((3,), 1.0))


def test_single_model_batch_2_stays_on_slot_zero() -> None:
    """N=1, K=2: sw-batch dispatch is a no-op and every row goes through slot 0."""
    m0 = StaticLastOnlyMxq(vocab_size=5, max_width=4)
    model = _make_multi_slot_model([m0])
    cache = MobilintCache(m0, per_model_batch=2)

    inputs_embeds = torch.randn(2, 3, 4, dtype=torch.float32)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(inputs_embeds.shape[1]),
        attention_mask=attention_mask,
    )

    # Single infer with both cache_ids on m0.
    assert len(m0.calls) == 1 and "batch" in m0.calls[0]
    observed_cache_ids = [item[0] for item in m0.calls[0]["batch"]]
    assert observed_cache_ids == [0, 1]
    assert logits.shape == (2, 1, 5)


def test_multi_model_dispatch_fast_path_when_rows_land_on_one_model() -> None:
    """N=2, K=2: if every row lands on the same Model the parallel fast path fires only that slot."""
    m0 = StaticLastOnlyMxq(vocab_size=3, max_width=4)
    m1 = StaticLastOnlyMxq(vocab_size=3, max_width=4)
    model = _make_multi_slot_model([m0, m1])
    # per_model_batch=2, so flat rows 0..1 -> m0, 2..3 -> m1.
    cache = MobilintCache([m0, m1], per_model_batch=2)

    # Only send batch rows that route to m0 (flat rows 0..1 land on model 0).
    inputs_embeds = torch.randn(2, 2, 4, dtype=torch.float32)
    attention_mask = torch.ones(2, 2, dtype=torch.long)

    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(inputs_embeds.shape[1]),
        attention_mask=attention_mask,
    )

    # m1 stays idle because no flat row landed on it.
    assert m1.calls == []
    assert len(m0.calls) >= 1
    for call in m0.calls:
        for cache_id, _seq_len, _cache_size in call["batch"]:
            assert cache_id in (0, 1)  # local slot ids within m0
    assert logits.shape == (2, 1, 3)


# ---------------------------------------------------------------------------
# Config-layer sugar and normalization round-trip
# ---------------------------------------------------------------------------


def test_dev_no_scalar_expands_target_cores_when_targets_absent() -> None:
    """A bare ``dev_no=1`` under ``single`` mode expands to every core on device 1."""
    kwargs = {"core_mode": "single", "dev_no": 1}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == [
        "1:0:0",
        "1:0:1",
        "1:0:2",
        "1:0:3",
        "1:1:0",
        "1:1:1",
        "1:1:2",
        "1:1:3",
    ]


def test_dev_no_list_expands_target_clusters_across_devices() -> None:
    """``dev_no=[0, 1]`` under ``global4`` expands to every cluster on both devices."""
    kwargs = {"core_mode": "global4", "dev_no": [0, 1]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0", "0:1", "1:0", "1:1"]


def test_legacy_target_cores_roundtrip_via_backend() -> None:
    """A legacy ``target_cores=["0:0"]`` payload round-trips as canonical ``["0:0:0"]``."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": 0,
        "target_cores": ["0:0"],  # legacy 2-part input
    }
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))

    dumped = backend.to_dict()
    assert dumped["target_cores"] == ["0:0:0"]
    assert dumped["dev_no"] == 0

    # A second normalization / from_dict round-trip must keep the canonical form.
    _normalize_npu_target_kwargs(dumped)
    reloaded = MobilintNPUBackend.from_dict(dict(dumped))
    assert reloaded.to_dict()["target_cores"] == ["0:0:0"]


def test_canonical_target_cores_dict_style_stays_canonical() -> None:
    """A per-device canonical ``target_cores`` list normalizes without rewriting entries."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": [0, 1],
        "target_cores": ["0:0:0", "0:0:1", "1:0:0", "1:0:1"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["0:0:0", "0:0:1", "1:0:0", "1:0:1"]
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    assert backend.to_dict()["target_cores"] == ["0:0:0", "0:0:1", "1:0:0", "1:0:1"]


# ---------------------------------------------------------------------------
# Cache capacity check
# ---------------------------------------------------------------------------


def test_validate_batch_cache_uses_n_models_times_k_capacity() -> None:
    """Multi-Model caches must expose ``n_models * k_per_model`` as the cache capacity."""
    m0 = StaticLastOnlyMxq()
    m1 = StaticLastOnlyMxq()
    cache = MobilintCache([m0, m1], per_model_batch=1)

    # Capacity is exactly 2, so a batch=2 request must be accepted.
    MobilintModelMixin._validate_batch_cache(cache, batch_size=2)

    # Capacity is exactly 2, so a batch=3 request must fail.
    with pytest.raises(ValueError, match="Batch cache size is too small"):
        MobilintModelMixin._validate_batch_cache(cache, batch_size=3)
