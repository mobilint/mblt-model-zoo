"""Regression tests for multi-slot batched cache class resolution.

Guards :func:`benchmark_utils._build_batched_mobilint_cache` against silently
falling back to plain :class:`MobilintCache` when a model declares a
specialized cache subclass via :meth:`MobilintModelMixin.get_mobilint_cache_cls`.

Motivating bug (PR #109 review): Qwen3-VL text ``llm_forward`` hard-fails on
non-:class:`MobilintDeepStackCache` ``past_key_values``, so the multi-slot
fake-prefill VLM decode benchmark tripped ``TypeError`` as soon as decode
measurement started.
"""

from __future__ import annotations

from types import SimpleNamespace

from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    _build_batched_mobilint_cache,
    _resolve_mobilint_cache_class,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import (
    MobilintCache,
    MobilintDeepStackCache,
)


class _PlainMultiSlotModel:
    """LLM stub whose backend hosts a two-slot Mobilint NPU backend."""

    def __init__(self, mxq_models: tuple[object, ...], k_per_model: int) -> None:
        self.npu_backend = SimpleNamespace(
            mxq_models=list(mxq_models),
            k_per_model=k_per_model,
        )


class _DeepStackMultiSlotModel(_PlainMultiSlotModel):
    """Qwen3-VL-style stub that declares :class:`MobilintDeepStackCache`."""

    @classmethod
    def get_mobilint_cache_cls(cls) -> type[MobilintDeepStackCache]:
        return MobilintDeepStackCache


def test_resolve_mobilint_cache_class_defaults_to_plain_cache() -> None:
    """A model without ``get_mobilint_cache_cls`` resolves to plain ``MobilintCache``."""
    model = _PlainMultiSlotModel(mxq_models=(object(),), k_per_model=1)

    resolved = _resolve_mobilint_cache_class(model)

    assert resolved is MobilintCache


def test_resolve_mobilint_cache_class_reads_model_classmethod() -> None:
    """A model overriding ``get_mobilint_cache_cls`` steers the resolver."""
    model = _DeepStackMultiSlotModel(mxq_models=(object(),), k_per_model=1)

    resolved = _resolve_mobilint_cache_class(model)

    assert resolved is MobilintDeepStackCache


def test_resolve_mobilint_cache_class_rejects_non_mobilint_cache_subclass() -> None:
    """A model exposing an unrelated class falls back to :class:`MobilintCache`.

    Guards the resolver against a stray override that returns something the
    downstream multi-slot builder cannot construct — the fallback keeps the
    benchmark path from crashing on model bugs and forces the guard to be
    obvious in code review rather than a runtime attribute error.
    """

    class _Rogue:
        @classmethod
        def get_mobilint_cache_cls(cls) -> type[object]:
            return object

    resolved = _resolve_mobilint_cache_class(_Rogue())

    assert resolved is MobilintCache


def test_build_batched_mobilint_cache_defaults_to_plain_cache_for_plain_model() -> None:
    """Plain LLM stubs receive a :class:`MobilintCache` from the multi-slot builder."""
    slot_a, slot_b = object(), object()
    model = _PlainMultiSlotModel(mxq_models=(slot_a, slot_b), k_per_model=1)

    cache = _build_batched_mobilint_cache(model, batch_size=2)

    assert isinstance(cache, MobilintCache)
    assert type(cache) is MobilintCache
    assert cache.mxq_models == [slot_a, slot_b]
    assert cache.k_per_model == 1


def test_build_batched_mobilint_cache_uses_declared_cache_class() -> None:
    """Qwen3-VL-style stubs receive a :class:`MobilintDeepStackCache` automatically.

    Regression for PR #109: the multi-slot builder previously always
    constructed a plain :class:`MobilintCache`, which the Qwen3-VL text
    decoder rejected with :class:`TypeError` during fake-prefill VLM decode
    benchmarks.
    """
    slot_a, slot_b = object(), object()
    model = _DeepStackMultiSlotModel(mxq_models=(slot_a, slot_b), k_per_model=1)

    cache = _build_batched_mobilint_cache(model, batch_size=2)

    assert isinstance(cache, MobilintDeepStackCache)
    assert cache.mxq_models == [slot_a, slot_b]
    assert cache.k_per_model == 1


def test_build_batched_mobilint_cache_explicit_cache_cls_overrides_model() -> None:
    """An explicit ``cache_cls`` argument wins over the model's declared class.

    Preserves the escape hatch for callers that need to build a specific
    cache subclass regardless of the model's default (e.g. tests exercising
    the plain :class:`MobilintCache` path against a Qwen3-VL stub).
    """
    slot_a, slot_b = object(), object()
    model = _DeepStackMultiSlotModel(mxq_models=(slot_a, slot_b), k_per_model=1)

    cache = _build_batched_mobilint_cache(model, batch_size=2, cache_cls=MobilintCache)

    assert type(cache) is MobilintCache
