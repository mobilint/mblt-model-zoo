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


@pytest.mark.parametrize(
    "target_device",
    ["regulus-ra", "regulus-rb", "regulus-ra-usb", "regulus-rb-usb"],
)
def test_target_device_setter_reassigns_backend_class_across_boards(
    target_device: str,
) -> None:
    """Rebuild the backend to the destination class when the setter switches boards.

    Reproduces the HF ``PretrainedConfig.from_dict`` flow: an Aries-shaped
    config (target_cores expanded to the 2×4 grid on ``__init__``) then
    receives a ``target_device`` override from a user ``from_pretrained``
    kwarg via ``setattr``. The setter must rebuild the backend on the
    destination class *and* reset the source board's topology fields,
    otherwise the Aries-shaped ``target_cores`` propagates into
    ``MobilintRegulusBackend`` and its single-core validator rejects the
    spec.
    """

    config = _TargetDeviceConfig()  # defaults to Aries, expands 8-core grid
    assert type(config.npu_backend).__name__ == "MobilintAriesBackend"
    assert len(config.to_dict()["target_cores"]) == 8

    config.target_device = target_device

    assert type(config.npu_backend).__name__ == "MobilintRegulusBackend"
    assert config.target_device == target_device
    assert config.to_dict()["target_cores"] == ["0:0:0"]


def test_target_device_setter_is_a_noop_when_the_class_does_not_change() -> None:
    """Preserve the topology fields when the caller reassigns to the same board class."""

    config = _TargetDeviceConfig()
    original_backend = config.npu_backend
    original_cores = list(config.to_dict()["target_cores"])

    # ``aries`` normalizes to ``aries-rb`` — same class as the default, so
    # the setter must not rebuild the backend and must preserve the
    # already-expanded topology.
    config.target_device = "aries"

    assert config.npu_backend is original_backend
    assert config.to_dict()["target_cores"] == original_cores
