"""Compatibility helpers for Qwen3-VL transformer tests."""

from __future__ import annotations

import importlib.util

import pytest

_QWEN3_VL_MODULES = (
    "transformers.models.qwen3_vl.configuration_qwen3_vl",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen3_vl.processing_qwen3_vl",
)

_QWEN3_VL_SKIP_REASON = (
    "Installed transformers does not provide the upstream Qwen3-VL classes "
    "(requires transformers>=4.57.0)."
)


def transformers_supports_qwen3_vl() -> bool:
    """Return whether the installed Transformers exposes the upstream Qwen3-VL modules."""
    for module_name in _QWEN3_VL_MODULES:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except ModuleNotFoundError:
            return False

    return True


def skip_if_transformers_lacks_qwen3_vl_support() -> None:
    """Skip the current module when upstream Qwen3-VL support is unavailable."""
    if transformers_supports_qwen3_vl():
        return

    pytest.skip(_QWEN3_VL_SKIP_REASON, allow_module_level=True)


def _transformers_version_tuple() -> tuple[int, int]:
    import transformers

    parts = transformers.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


QWEN3_VL_PIPELINE_TF_5_4_REGRESSION_REASON = (
    "transformers>=5.4 broke the image-text-to-text pipeline / Qwen3-VL "
    "generate integration: ``_prepare_model_inputs`` returns ``inputs_tensor`` "
    "as a Python list, which raises ``AttributeError: 'list' object has no "
    "attribute 'shape'`` at ``batch_size = inputs_tensor.shape[0]``. Skip until "
    "either upstream restores tensor shape or the mblt_model_zoo wrapper "
    "compensates."
)


def skip_qwen3_vl_pipeline_if_tf_5_4() -> None:
    """Module-level skip for Qwen3-VL pipeline tests on transformers>=5.4."""
    if _transformers_version_tuple() >= (5, 4):
        pytest.skip(QWEN3_VL_PIPELINE_TF_5_4_REGRESSION_REASON, allow_module_level=True)


QWEN3_VL_8B_MXQ_INCOMPATIBLE_REASON = (
    "Qwen3-VL-8B MXQ (regular and Batch16) hits 'NPU-only model output order "
    "mismatch' with qbruntime 1.4.0 / mblt_npu 0.1.0 and access-violation-"
    "crashes at qbruntime.Model.__init__. Re-enable once an updated MXQ ships "
    "that matches the current runtime output ordering."
)


def skip_qwen3_vl_8b_module() -> None:
    """Module-level skip for 8B-only Qwen3-VL test files."""
    pytest.skip(QWEN3_VL_8B_MXQ_INCOMPATIBLE_REASON, allow_module_level=True)


def skip_if_static_vision(pipe, feature: str) -> None:
    """Skip the current test when the pipeline's processor is static-vision.

    Multi-image and video inputs require a dynamic-vision Qwen3-VL release
    (3-input vision MXQ with per-image / per-frame 2D RoPE in the text
    decoder). Static releases hard-fail such inputs in the processor, so
    tests exercising those code paths must skip on those model variants.
    """
    processor = getattr(pipe, "processor", None)
    if processor is None or getattr(processor, "dynamic_vision", True):
        return

    pytest.skip(
        f"{feature} requires a dynamic-vision Qwen3-VL release; loaded "
        f"processor is in static mode (dynamic_vision=False)."
    )
