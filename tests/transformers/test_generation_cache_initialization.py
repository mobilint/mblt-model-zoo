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


def test_eagle3_generation_cache_initializes_when_cache_attribute_is_none() -> None:
    """Initialize the EAGLE-3 cache when its inherited default is empty."""
    model = _DummyEagle3GenerationModel()

    cache = model._get_cache("mobilint-eagle3", batch_size=1, max_cache_len=1)

    assert isinstance(cache, MobilintEagle3Cache)
    assert (cache.base_mxq_model, cache.draft_mxq_model) == model.mxq_models
