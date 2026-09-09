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


def test_npu_prefill_chunk_size_kwarg_survives_from_dict() -> None:
    """Keep the caller's ``npu_prefill_chunk_size`` override across ``from_dict``.

    ``MobilintNPUBackend.from_dict`` does not consume
    ``npu_prefill_chunk_size`` — it is a config-only attribute stored
    on ``self.__dict__`` and read by
    ``resolve_npu_prefill_chunk_size`` as a fallback / per-core-mode
    mapping. Including it in ``_NPU_BACKEND_KWARG_FIELDS`` (an earlier
    revision did) caused ``_pop_consumed_backend_kwargs`` to delete it
    before HF's ``PretrainedConfig.__init__`` could route it through
    the property setter, so the config surfaced ``None`` and the
    resolver silently fell back to 128.
    """

    config = _TargetDeviceConfig.from_dict(
        {"model_type": _TargetDeviceConfig.model_type},
        npu_prefill_chunk_size={"single": 64, "global4": 96, "global8": 192},
    )
    assert config.npu_prefill_chunk_size == {"single": 64, "global4": 96, "global8": 192}


def test_from_dict_buffers_target_device_across_topology_probe_hazard() -> None:
    """Apply cross-board overrides after the ``super().from_dict`` kwargs loop.

    HF ``PretrainedConfig.from_dict`` probes every override with
    ``hasattr`` before ``setattr``. The topology getters
    (``target_cores`` / ``target_clusters`` / ``core_mode`` /
    ``dev_no``) all trigger a spec finalize on the current pending
    state, so a caller override that combines ``target_device`` with a
    board-specific mode (e.g. ``core_mode="global8"``) raises during
    the probe against the source board's topology before the target-
    device rebuild has a chance to run. Buffering the NPU kwargs and
    replaying them after ``super().from_dict`` returns lets the
    complete override set land on the destination board atomically.
    """

    config = _TargetDeviceConfig.from_dict(
        {"model_type": _TargetDeviceConfig.model_type, "target_device": "regulus-rb-usb"},
        core_mode="global8",
        target_clusters=[0, 1],
        target_device="aries-rb",
    )

    assert type(config.npu_backend).__name__ == "MobilintAriesBackend"
    assert config.target_device == "aries-rb"
    assert config.core_mode == "global8"
    assert set(config.to_dict()["target_clusters"]) == {"0:0", "0:1"}


@pytest.mark.parametrize("prefix", ["encoder", "decoder"])
def test_qwen3_asr_ctor_routes_revision_and_commit_hash_to_backend(prefix: str) -> None:
    """Forward ``revision`` / ``commit_hash`` kwargs to the nested NPU backend.

    ``MobilintConfigMixin`` exposes no forwarding property for
    ``revision`` or ``commit_hash``, so routing every prefixed kwarg
    through the sub-config setter would land those two on
    ``config.__dict__`` while leaving ``npu_backend.revision`` and
    ``npu_backend._commit_hash`` at their init-time defaults. Remote
    MXQ resolution then falls back to the shipped revision and can
    load the wrong artifact. Route ``revision`` and ``commit_hash``
    directly to the backend (mapping ``commit_hash`` to the internal
    ``_commit_hash`` attribute) while keeping the target-device rebuild
    routing intact for the fields that do have a config property.
    """

    pytest.importorskip("qwen_asr")
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.configuration_qwen3_asr import (
        MobilintQwen3ASRConfig,
    )

    prefix_ = f"{prefix}_"
    inner_config = "audio_config" if prefix == "encoder" else "text_config"

    config = MobilintQwen3ASRConfig(
        **{
            f"{prefix_}revision": "test-release-1",
            f"{prefix_}commit_hash": "0123456789abcdef",
        }
    )

    sub_backend = getattr(config.thinker_config, inner_config).npu_backend
    assert sub_backend.revision == "test-release-1"
    assert sub_backend._commit_hash == "0123456789abcdef"


@pytest.mark.parametrize("prefix", ["encoder", "decoder"])
def test_qwen3_asr_from_dict_routes_revision_and_commit_hash_to_backend(prefix: str) -> None:
    """Forward buffered ``revision`` / ``commit_hash`` kwargs to the nested backend.

    The Qwen3-ASR ``from_dict`` override buffers every ``encoder_*`` /
    ``decoder_*`` NPU kwarg to sidestep HF's hasattr-probe hazard and
    replays them via ``_apply_npu_backend_kwargs``, which drives the
    top-level config's setattr path. That path only reaches the nested
    backend for fields with a forwarding property on
    ``MobilintConfigMixin``. ``revision`` and ``commit_hash`` have no
    such property, so the same direct-to-backend routing that the
    constructor uses must apply on the ``from_dict`` path as well —
    otherwise remote MXQ resolution silently keeps the shipped
    revision.
    """

    pytest.importorskip("qwen_asr")
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.configuration_qwen3_asr import (
        MobilintQwen3ASRConfig,
    )

    prefix_ = f"{prefix}_"
    inner_config = "audio_config" if prefix == "encoder" else "text_config"

    config = MobilintQwen3ASRConfig.from_dict(
        {"model_type": MobilintQwen3ASRConfig.model_type},
        **{
            f"{prefix_}revision": "from-dict-release",
            f"{prefix_}commit_hash": "fedcba9876543210",
        },
    )

    sub_backend = getattr(config.thinker_config, inner_config).npu_backend
    assert sub_backend.revision == "from-dict-release"
    assert sub_backend._commit_hash == "fedcba9876543210"


@pytest.mark.parametrize("prefix", ["encoder", "decoder"])
def test_qwen3_asr_from_dict_buffers_prefixed_target_device_atomically(prefix: str) -> None:
    """Buffer ``encoder_*`` / ``decoder_*`` overrides on the Qwen3-ASR facade.

    ``MobilintQwen3ASRConfig`` does not inherit
    ``MobilintEncoderDecoderConfigMixin``, so the mixin's buffered
    ``from_dict`` does not apply. The facade's own ``from_dict`` must
    mirror the buffering pattern; otherwise HF's ``hasattr`` probe on
    e.g. ``encoder_target_clusters`` fires against a Regulus baseline
    that already has ``core_mode="global8"`` pending and raises before
    the ``encoder_target_device`` setter can rebuild to Aries.
    """

    pytest.importorskip("qwen_asr")
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.configuration_qwen3_asr import (
        MobilintQwen3ASRConfig,
    )

    prefix_ = f"{prefix}_"
    inner_config = "audio_config" if prefix == "encoder" else "text_config"
    inner_model_type = (
        "mobilint-qwen3_asr_audio_encoder" if prefix == "encoder"
        else "mobilint-qwen3_asr_text"
    )

    config = MobilintQwen3ASRConfig.from_dict(
        {
            "model_type": MobilintQwen3ASRConfig.model_type,
            "thinker_config": {
                inner_config: {
                    "model_type": inner_model_type,
                    "target_device": "regulus-rb-usb",
                }
            },
        },
        **{
            f"{prefix_}core_mode": "global8",
            f"{prefix_}target_clusters": [0, 1],
            f"{prefix_}target_device": "aries-rb",
        },
    )

    sub_config = getattr(config.thinker_config, inner_config)
    sub_backend = sub_config.npu_backend
    assert type(sub_backend).__name__ == "MobilintAriesBackend"
    assert getattr(config, f"{prefix_}core_mode") == "global8"
    assert set(sub_config.to_dict()["target_clusters"]) == {"0:0", "0:1"}


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
