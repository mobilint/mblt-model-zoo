"""Tests for Model Zoo target-device forwarding to mblt-npu-python."""

from __future__ import annotations

import pytest

from mblt_model_zoo.hf_transformers.utils.configuration_utils import (
    MobilintConfigMixin,
    MobilintEagle3ConfigMixin,
    MobilintEncoderDecoderConfigMixin,
)


class _TargetDeviceConfig(MobilintConfigMixin):
    model_type = "target-device-test"


class _EncoderDecoderTargetDeviceConfig(MobilintEncoderDecoderConfigMixin):
    model_type = "target-device-encdec-test"


class _Eagle3TargetDeviceConfig(MobilintEagle3ConfigMixin):
    model_type = "target-device-eagle3-test"


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


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("aries", "aries-rb"), ("regulus", "regulus-ra")],
)
def test_target_device_setter_normalizes_legacy_aliases(alias: str, canonical: str) -> None:
    """Rewrite legacy family aliases to their canonical board identifier.

    ``qbruntime>=1.4`` expects the canonical board name as the first
    positional argument to ``Accelerator``; without normalization the
    same-class fast path would leave the alias in place on the backend and
    surface at inference time as a runtime error from ``qbruntime`` for an
    unknown target device.
    """

    config = _TargetDeviceConfig(target_device=canonical)  # same class as alias
    config.target_device = alias

    assert config.target_device == canonical
    assert config.to_dict()["target_device"] == canonical


@pytest.mark.parametrize(
    "prefix",
    ["base", "draft", "fc"],
)
def test_eagle3_prefixed_target_device_setter_rebuilds_backend(prefix: str) -> None:
    """Rebuild the prefixed Eagle3 backend when its target_device setter switches boards.

    Every ``MobilintEagle3ConfigMixin`` sub-backend (``base_*``, ``draft_*``,
    ``fc_*``) exposes its own ``*_target_device`` property that HF's
    ``from_dict`` kwargs loop applies via ``setattr``. Each setter must run
    through the same cross-board rebuild logic as the single-backend mixin
    so a caller override actually reaches the destination board's class.
    """

    config = _Eagle3TargetDeviceConfig()
    backend_attr = f"{prefix}_npu_backend"
    setattr_key = f"{prefix}_target_device"
    original_backend = getattr(config, backend_attr)
    assert type(original_backend).__name__ == "MobilintAriesBackend"

    setattr(config, setattr_key, "regulus-rb-usb")

    rebuilt = getattr(config, backend_attr)
    assert type(rebuilt).__name__ == "MobilintRegulusBackend"
    assert rebuilt.target_device == "regulus-rb-usb"
    # Topology reset: the destination class fills its single-core default.
    assert config.to_dict()[f"{prefix}_target_cores"] == ["0:0:0"]


def test_target_device_setter_reapplies_pending_topology_atomically() -> None:
    """Preserve caller topology overrides across a cross-board setter switch.

    HF ``PretrainedConfig.from_dict`` applies caller kwargs via ``setattr``
    in insertion order, so ``core_mode="global8"`` may land on a Regulus
    baseline before the accompanying ``target_device="aries-rb"`` kwarg.
    The Regulus board rejects ``global8``; the previous rebuild
    serialized the source via ``to_dict`` and thus finalized that
    pending override against the wrong topology and raised. The atomic
    rebuild reads only board-agnostic raw attributes from the source
    backend and replays the caller's pending topology overrides
    (``dev_no`` / ``core_mode`` / ``target_cores`` / ``target_clusters``)
    onto the fresh destination pending accumulator, so the same override
    set behaves the same regardless of setattr order.
    """

    config = _TargetDeviceConfig(target_device="regulus-rb-usb")
    # Setter A: pending ``core_mode="global8"`` recorded against Regulus
    # (not finalized yet — Regulus would reject it if finalize ran).
    config.core_mode = "global8"
    # Setter B: cross-board rebuild to Aries. Must not finalize the
    # source pending; must replay ``core_mode="global8"`` on the fresh
    # Aries backend so the final state honours the caller's intent.
    config.target_device = "aries-rb"

    assert type(config.npu_backend).__name__ == "MobilintAriesBackend"
    assert config.target_device == "aries-rb"
    assert config.core_mode == "global8"
    # Aries's ``global8`` sugar covers both clusters.
    assert set(config.to_dict()["target_clusters"]) == {"0:0", "0:1"}


@pytest.mark.parametrize(
    "prefix",
    ["encoder", "decoder"],
)
def test_encoder_decoder_target_device_property_rebuilds_backend(prefix: str) -> None:
    """Expose ``encoder_target_device`` / ``decoder_target_device`` and rebuild across boards.

    ``MobilintEncoderDecoderConfigMixin`` previously offered no
    ``encoder_target_device`` / ``decoder_target_device`` property, so HF's
    ``from_dict`` kwargs loop silently dropped those keys via its
    ``hasattr`` gate. Advertising the setters — and routing them through
    the shared rebuild helper — makes prefixed board overrides work
    end-to-end for encoder-decoder models.
    """

    config = _EncoderDecoderTargetDeviceConfig()
    backend_attr = f"{prefix}_npu_backend"
    setattr_key = f"{prefix}_target_device"
    assert type(getattr(config, backend_attr)).__name__ == "MobilintAriesBackend"

    setattr(config, setattr_key, "regulus-rb-usb")

    rebuilt = getattr(config, backend_attr)
    assert type(rebuilt).__name__ == "MobilintRegulusBackend"
    assert getattr(config, setattr_key) == "regulus-rb-usb"
    assert config.to_dict()[f"{prefix}_target_cores"] == ["0:0:0"]
