"""Tests for the Qwen3-VL ``dynamic_vision`` config field and its downstream wiring."""

from __future__ import annotations

import logging

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLVisionConfig,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLVisionModel,
)


def test_vision_config_dynamic_vision_default_false() -> None:
    """Preserve backward compatibility: dynamic_vision defaults to False."""
    config = MobilintQwen3VLVisionConfig()
    assert config.dynamic_vision is False


def test_vision_config_dynamic_vision_round_trip() -> None:
    """Preserve ``dynamic_vision`` through ``to_dict`` / ``from_dict`` round-trip."""
    config = MobilintQwen3VLVisionConfig(dynamic_vision=True)
    assert config.dynamic_vision is True

    payload = config.to_dict()
    assert payload["dynamic_vision"] is True

    restored = MobilintQwen3VLVisionConfig.from_dict(payload)
    assert restored.dynamic_vision is True


def test_resolve_dynamic_vision_flag_static_matches_config() -> None:
    """No warning when a 1-input MXQ pairs with ``dynamic_vision=False``."""
    config = MobilintQwen3VLVisionConfig(dynamic_vision=False)
    assert MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(1, config) is False


def test_resolve_dynamic_vision_flag_dynamic_matches_config() -> None:
    """No warning when a 3-input MXQ pairs with ``dynamic_vision=True``."""
    config = MobilintQwen3VLVisionConfig(dynamic_vision=True)
    assert MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(3, config) is True


def test_resolve_dynamic_vision_flag_warns_on_config_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn (and trust MXQ) when the config disagrees with the compiled input count."""
    config = MobilintQwen3VLVisionConfig(dynamic_vision=False)
    with caplog.at_level(logging.WARNING, logger="mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl"):
        detected = MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(3, config)
    assert detected is True
    assert any("disagrees with vision MXQ detection" in rec.message for rec in caplog.records)


def test_resolve_dynamic_vision_flag_rejects_unknown_input_count() -> None:
    """Reject unrecognized MXQ signatures rather than guessing."""
    config = MobilintQwen3VLVisionConfig()
    with pytest.raises(ValueError, match="1 \\(static\\) or 3 \\(dynamic"):
        MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(2, config)
