"""Tests for Model Zoo target-device forwarding to mblt-npu-python."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.utils.configuration_utils import MobilintConfigMixin


class _TargetDeviceConfig(MobilintConfigMixin):
    model_type = "target-device-test"


def test_transformers_config_defaults_to_aries_rb() -> None:
    """Construct the default backend through the board-aware shared package."""

    config = _TargetDeviceConfig()

    assert type(config.npu_backend).__name__ == "MobilintAriesBackend"
    assert config.to_dict()["target_device"] == "aries-rb"


def test_transformers_config_selects_regulus_backend() -> None:
    """Forward an explicit Regulus board setting without direct class usage."""

    config = _TargetDeviceConfig(target_device="regulus-rb")

    assert type(config.npu_backend).__name__ == "MobilintRegulusBackend"
    assert config.to_dict()["target_device"] == "regulus-rb"
