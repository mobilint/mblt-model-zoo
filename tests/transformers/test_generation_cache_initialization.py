"""Regression tests for generation cache initialization."""

from types import SimpleNamespace

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache, MobilintEagle3Cache
from mblt_model_zoo.hf_transformers.utils.generation_utils import (
    MobilintEagle3GenerationMixin,
    MobilintGenerationMixin,
)


class _DummyGenerationModel(MobilintGenerationMixin):
    """Minimal generation model with the upstream-style empty cache default."""

    _cache = None

    def __init__(self) -> None:
        self.config = SimpleNamespace(max_batch_size=1)
        self.mxq_model = object()

    def get_cache_mxq_model(self) -> object:
        """Return a cache backend stub."""
        return self.mxq_model


class _MultiSlotGenerationModel(MobilintGenerationMixin):
    """Generation model backed by a multi-slot ``npu_backend`` stub."""

    _cache = None

    def __init__(self, mxq_models: tuple[object, ...], k_per_model: int) -> None:
        aggregate_batch = len(mxq_models) * k_per_model
        self.config = SimpleNamespace(max_batch_size=aggregate_batch)
        self.npu_backend = SimpleNamespace(
            mxq_models=list(mxq_models),
            k_per_model=k_per_model,
        )


class _DummyEagle3GenerationModel(MobilintEagle3GenerationMixin):
    """Minimal EAGLE-3 model with the upstream-style empty cache default."""

    _cache = None

    def __init__(self) -> None:
        self.mxq_models = (object(), object())

    def get_cache_mxq_models(self) -> tuple[object, object]:
        """Return base and draft cache backend stubs."""
        return self.mxq_models


def test_generation_cache_initializes_when_cache_attribute_is_none() -> None:
    """Initialize rather than resetting a class-level empty cache."""
    model = _DummyGenerationModel()

    cache = model._get_cache("mobilint", batch_size=1, max_cache_len=1)

    assert isinstance(cache, MobilintCache)
    assert cache.mxq_model is model.mxq_model


def test_generation_cache_routes_across_multi_slot_backend() -> None:
    """``_get_cache`` must build a multi-slot cache when the backend hosts ``N > 1`` Models.

    Regression: the legacy signature routed every row to slot 0 via
    ``MobilintCache(slot_0_model, batch_size=B)``, breaking a non-batch MXQ
    (``K == 1``) with ``B > 1`` because ``cache_id > 0`` targeted a slot the
    backend never allocated.
    """
    slot_a, slot_b = object(), object()
    model = _MultiSlotGenerationModel(mxq_models=(slot_a, slot_b), k_per_model=1)

    cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)

    assert isinstance(cache, MobilintCache)
    assert cache.mxq_models == [slot_a, slot_b]
    assert cache.n_models == 2
    assert cache.k_per_model == 1
    assert cache.batch_size == 2
    # ``slot_of`` must route flat rows to their owning Model.
    assert cache.slot_of(0) == (0, 0)
    assert cache.slot_of(1) == (1, 0)
    assert cache.model_of(0) is slot_a
    assert cache.model_of(1) is slot_b


def test_generation_cache_preserves_hardware_batch_on_single_slot_backend() -> None:
    """A single-Model backend with ``K > 1`` keeps the hardware-batch shape (``N=1``)."""
    only_slot = object()
    model = _MultiSlotGenerationModel(mxq_models=(only_slot,), k_per_model=4)

    cache = model._get_cache("mobilint", batch_size=1, max_cache_len=1)

    assert isinstance(cache, MobilintCache)
    assert cache.n_models == 1
    assert cache.k_per_model == 4
    assert cache.batch_size == 4
    assert cache.mxq_model is only_slot


def test_eagle3_generation_cache_initializes_when_cache_attribute_is_none() -> None:
    """Initialize the EAGLE-3 cache when its inherited default is empty."""
    model = _DummyEagle3GenerationModel()

    cache = model._get_cache("mobilint-eagle3", batch_size=1, max_cache_len=1)

    assert isinstance(cache, MobilintEagle3Cache)
    assert (cache.base_mxq_model, cache.draft_mxq_model) == model.mxq_models


def test_generation_cache_rebuilds_when_backend_topology_changes() -> None:
    """Reuse must invalidate on ``(mxq_models, k_per_model)`` change even at equal capacity.

    Regression for PR #109 review: with an ``N=2, K=2`` backend disposed and
    re-created as ``N=4, K=1``, the aggregate row capacity stays at ``4`` but
    every ``(model_idx, cache_id)`` mapping changes. ``slot_of`` on the stale
    cache would still assume ``K=2`` and return an invalid ``cache_id`` for
    slots ``1..N-1``, either erroring or silently pinning dispatch to slot 0.
    """
    initial_slots = (object(), object())
    model = _MultiSlotGenerationModel(mxq_models=initial_slots, k_per_model=2)

    first_cache = model._get_cache("mobilint", batch_size=4, max_cache_len=1)

    assert first_cache is model._cache
    assert first_cache.n_models == 2
    assert first_cache.k_per_model == 2
    assert first_cache.batch_size == 4

    # Dispose+recreate the backend into a different topology with matching total
    # capacity: N=4 slots, K=1 per model.
    new_slots = (object(), object(), object(), object())
    model.npu_backend = SimpleNamespace(mxq_models=list(new_slots), k_per_model=1)

    second_cache = model._get_cache("mobilint", batch_size=4, max_cache_len=1)

    assert second_cache is not first_cache
    assert second_cache is model._cache
    assert second_cache.mxq_models == list(new_slots)
    assert second_cache.n_models == 4
    assert second_cache.k_per_model == 1
    assert second_cache.batch_size == 4
    assert second_cache.slot_of(0) == (0, 0)
    assert second_cache.slot_of(3) == (3, 0)
    assert second_cache.model_of(3) is new_slots[3]


def test_generation_cache_rebuilds_when_slot_handles_change() -> None:
    """Reuse must invalidate when a slot's ``qbruntime.Model`` handle is swapped.

    A relaunch that preserves ``(N, K)`` but hands out fresh ``qbruntime.Model``
    handles is functionally a new backend: the cache's cached handles point at
    disposed slots. Rebuild rather than reset in place.
    """
    initial_slots = (object(), object())
    model = _MultiSlotGenerationModel(mxq_models=initial_slots, k_per_model=1)

    first_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)

    # Same (N, K), but new Model handles.
    new_slots = (object(), object())
    model.npu_backend = SimpleNamespace(mxq_models=list(new_slots), k_per_model=1)

    second_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)

    assert second_cache is not first_cache
    assert second_cache.mxq_models == list(new_slots)


def test_generation_cache_resets_when_topology_matches() -> None:
    """Reuse the existing cache when both slot handles and ``k_per_model`` match."""
    slots = (object(), object())
    model = _MultiSlotGenerationModel(mxq_models=slots, k_per_model=2)

    first_cache = model._get_cache("mobilint", batch_size=4, max_cache_len=1)
    first_cache.layers[0].set_seq_length(5)

    second_cache = model._get_cache("mobilint", batch_size=4, max_cache_len=1)

    assert second_cache is first_cache
    assert second_cache.layers[0].get_seq_length() == 0
