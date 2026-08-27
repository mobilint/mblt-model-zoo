"""Regression tests for MobilintVisionTextConfigMixin.from_dict kwargs ordering.

Upstream ``PretrainedConfig.from_dict`` iterates ``kwargs`` and calls
``hasattr(config, key)`` for each override. On composite configs whose
prefixed NPU properties route through sub-config backends, the getter probe
triggers the backend's lazy finalize on whatever pending state is visible
at that moment. If ``config_dict`` encodes a ``global8`` default and the
caller narrows target_cores to a single cluster, the finalize that fires
before the matching ``core_mode='single'`` override lands trips
``_validate_global8_coverage``. The fix buffers the sub-backend keys until
after upstream returns and applies them as one group.
"""

from __future__ import annotations

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLConfig,
)


def _global8_payload() -> dict:
    """Build a config_dict pinned to global8 covering both clusters of device 0."""
    base = MobilintQwen3VLConfig(
        vision_core_mode="global8",
        text_core_mode="global8",
        vision_target_clusters=[0, 1],
        text_target_clusters=[0, 1],
    )
    return base.to_dict()


def test_from_dict_narrows_global8_default_to_single_cluster_via_prefixed_kwargs() -> None:
    """A caller narrowing target_cores to one cluster and switching to
    ``single`` mode must succeed even when the config_dict defaults to
    ``global8`` covering both clusters."""
    payload = _global8_payload()

    restored = MobilintQwen3VLConfig.from_dict(
        payload,
        vision_target_cores=["0:1:0"],
        text_target_cores=["0:1:1"],
        vision_core_mode="single",
        text_core_mode="single",
    )

    assert restored.vision_core_mode == "single"
    assert restored.text_core_mode == "single"
    assert restored.vision_target_cores == ["0:1:0"]
    assert restored.text_target_cores == ["0:1:1"]


def test_from_dict_target_cluster_override_reaches_setter_under_global8() -> None:
    """A ``vision_target_clusters`` override consistent with ``global8`` must
    still round-trip through the sub-backend setter — guards against the fix
    silently dropping keys instead of routing them through
    ``_apply_sub_backend_kwargs``."""
    payload = _global8_payload()

    restored = MobilintQwen3VLConfig.from_dict(
        payload,
        vision_target_clusters=[0, 1],
        text_target_clusters=[0, 1],
    )

    assert restored.vision_core_mode == "global8"
    assert restored.text_core_mode == "global8"
    # global8 canonicalizes to per-device cluster identifiers rather than the
    # bare cluster indices we passed in; observing the canonical form proves
    # our override reached the sub-config setter and re-finalized the spec.
    assert restored.vision_target_clusters == ["0:0", "0:1"]
    assert restored.text_target_clusters == ["0:0", "0:1"]


def test_from_dict_leaves_non_sub_backend_kwargs_in_unused() -> None:
    """Non NPU sub-backend kwargs must still round-trip through upstream's
    unused_kwargs channel. The fix pops only the ``_SUB_BACKEND_FIELDS``
    prefixed keys; every other kwarg stays visible to upstream."""
    payload = _global8_payload()

    _, unused = MobilintQwen3VLConfig.from_dict(
        payload,
        vision_core_mode="single",
        vision_target_cores=["0:1:0"],
        text_core_mode="single",
        text_target_cores=["0:1:1"],
        some_unknown_kwarg="preserved",
        return_unused_kwargs=True,
    )

    assert unused.get("some_unknown_kwarg") == "preserved"
    for consumed in (
        "vision_core_mode",
        "vision_target_cores",
        "text_core_mode",
        "text_target_cores",
    ):
        assert consumed not in unused
