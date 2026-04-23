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
