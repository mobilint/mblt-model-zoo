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
from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLProcessor,
    MobilintQwen3VLVideoProcessor,
)


class _VisionStub:
    def __init__(self, uses_dynamic_vision: bool) -> None:
        self._uses_dynamic_vision = uses_dynamic_vision


class _ModelStub:
    """Mimic the Qwen3-VL model attribute layout expected by the sync helper."""

    def __init__(self, uses_dynamic_vision: bool, *, nested: bool = True) -> None:
        vision = _VisionStub(uses_dynamic_vision)
        if nested:
            inner = type("_Inner", (), {"visual": vision})()
            self.model = inner
        else:
            self.visual = vision


def _make_processor(prior_dynamic_vision: object = "unset") -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``.

    ``prior_dynamic_vision="unset"`` leaves the class-level default untouched;
    otherwise the given value is written into the instance ``__dict__`` to
    simulate a config-derived or user-explicit assignment.
    """
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    if prior_dynamic_vision != "unset":
        proc.dynamic_vision = prior_dynamic_vision
    return proc


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


def test_sync_dynamic_vision_from_model_upgrades_default_to_true() -> None:
    """Class default (False) upgrades to detected True with no prior assignment."""
    proc = _make_processor()
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True))
    assert proc.dynamic_vision is True
    assert proc.video_processor.dynamic_vision is True


def test_sync_dynamic_vision_from_model_overwrites_config_derived_value() -> None:
    """A config-derived ``dynamic_vision=False`` is silently replaced by detected True.

    This is the regression the fix addresses: ``from_pretrained`` always writes a
    config-derived value into ``__dict__``, so any lingering mismatch guard
    would raise here even though the caller invoked the helper precisely to
    resolve that mismatch.
    """
    proc = _make_processor(prior_dynamic_vision=False)
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True))
    assert proc.dynamic_vision is True
    assert proc.video_processor.dynamic_vision is True


def test_sync_dynamic_vision_from_model_overwrites_user_override() -> None:
    """An explicit user override also loses to the detected value."""
    proc = _make_processor(prior_dynamic_vision=True)
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=False))
    assert proc.dynamic_vision is False
    assert proc.video_processor.dynamic_vision is False


def test_sync_dynamic_vision_from_model_accepts_flat_model_layout() -> None:
    """Some Qwen3-VL variants expose ``visual`` directly, not under ``.model``."""
    proc = _make_processor()
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True, nested=False))
    assert proc.dynamic_vision is True


def test_sync_dynamic_vision_from_model_rejects_non_qwen3_vl_model() -> None:
    """A model whose vision submodule lacks ``_uses_dynamic_vision`` is rejected."""
    proc = _make_processor()

    class _AlienModel:
        model = type("_Inner", (), {"visual": object()})()

    with pytest.raises(ValueError, match="_uses_dynamic_vision"):
        proc.sync_dynamic_vision_from_model(_AlienModel())
