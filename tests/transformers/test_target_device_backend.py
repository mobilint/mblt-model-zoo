"""Tests for Model Zoo target-device forwarding to mblt-npu-python."""

from __future__ import annotations

import pytest

from mblt_model_zoo.hf_transformers.utils.configuration_utils import MobilintConfigMixin


class _TargetDeviceConfig(MobilintConfigMixin):
    model_type = "target-device-test"


def test_transformers_config_defaults_to_aries_rb() -> None:
    """Construct the default backend through the board-aware shared package."""

    config = _TargetDeviceConfig()

    assert type(config.npu_backend).__name__ == "MobilintAriesBackend"
    assert config.to_dict()["target_device"] == "aries-rb"


@pytest.mark.parametrize(
    ("target_device", "backend_class_name"),
    [
        ("aries-rb", "MobilintAriesBackend"),
        ("regulus-ra", "MobilintRegulusBackend"),
        ("regulus-rb", "MobilintRegulusBackend"),
        ("regulus-rb-usb", "MobilintRegulusBackend"),
    ],
)
def test_transformers_config_forwards_target_device(
    target_device: str, backend_class_name: str
) -> None:
    """Forward every documented Model Zoo target device without direct class usage.

    ``regulus-rb-usb`` reaches ``MobilintRegulusBackend`` via mblt-npu-python's
    dispatch table; it requires the qbruntime 1.4 shared package. The other
    boards remain covered so a regression at the forwarding layer surfaces on
    Aries, Regulus PCIe, and Regulus USB alike.
    """

    config = _TargetDeviceConfig(target_device=target_device)

    assert type(config.npu_backend).__name__ == backend_class_name
    assert config.to_dict()["target_device"] == target_device
