"""Tests for TPS CLI device-default resolution when only subconfig MXQ paths are given.

The TPS CLI treats any MXQ artifact as a signal to prefer CPU/NPU defaults, but the
subconfig-scoped ``--vision-mxq-path`` / ``--text-mxq-path`` / EAGLE-3 prefix paths must
count the same as the base ``--mxq-path`` so a non-``mobilint/`` model still gets the
NPU-oriented defaults without extra flags.
"""

from __future__ import annotations

import argparse

import pytest

from mblt_model_zoo.cli import tps as tps_cli


_NON_MOBILINT_MODEL = "someorg/some-model"


def _base_args(**overrides: object) -> argparse.Namespace:
    """Return an argparse Namespace shaped like the TPS CLI after argparse runs."""
    ns = argparse.Namespace(
        model=_NON_MOBILINT_MODEL,
        device=None,
        device_backend=None,
        mxq_path=None,
        base_mxq_path=None,
        draft_mxq_path=None,
        fc_mxq_path=None,
        vision_mxq_path=None,
        text_mxq_path=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_vision_mxq_path_triggers_npu_defaults_without_base_mxq_path() -> None:
    """Only ``--vision-mxq-path`` on a non-mobilint model should still pick CPU/NPU."""
    args = _base_args(vision_mxq_path="/tmp/vision.mxq")

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cpu"
    assert args.device_backend == "npu"


def test_text_mxq_path_triggers_npu_defaults_without_base_mxq_path() -> None:
    """Only ``--text-mxq-path`` on a non-mobilint model should still pick CPU/NPU."""
    args = _base_args(text_mxq_path="/tmp/text.mxq")

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cpu"
    assert args.device_backend == "npu"


@pytest.mark.parametrize("attr", ["base_mxq_path", "draft_mxq_path", "fc_mxq_path"])
def test_eagle3_prefix_mxq_path_triggers_npu_defaults(attr: str) -> None:
    """Any EAGLE-3 prefix MXQ path should also steer defaults toward CPU/NPU."""
    args = _base_args(**{attr: f"/tmp/{attr}.mxq"})

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cpu"
    assert args.device_backend == "npu"


def test_base_mxq_path_alone_still_triggers_npu_defaults() -> None:
    """Existing behavior: ``--mxq-path`` on a non-mobilint model resolves to CPU/NPU."""
    args = _base_args(mxq_path="/tmp/global.mxq")

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cpu"
    assert args.device_backend == "npu"


def test_no_mxq_path_on_non_mobilint_model_keeps_gpu_defaults() -> None:
    """Without any MXQ artifact, a non-mobilint model should default to cuda/gpu."""
    args = _base_args()

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cuda"
    assert args.device_backend == "gpu"


def test_explicit_device_and_backend_are_preserved_with_subconfig_mxq() -> None:
    """Explicit ``--device`` / ``--device-backend`` must win over the subconfig-derived default."""
    args = _base_args(
        vision_mxq_path="/tmp/vision.mxq",
        device="cuda:0",
        device_backend="gpu",
    )

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cuda:0"
    assert args.device_backend == "gpu"


def test_effective_mxq_path_prefers_base_over_subconfig() -> None:
    """When both base and subconfig MXQ paths are set, the base one is used for the decision."""
    args = _base_args(
        mxq_path="/tmp/global.mxq",
        vision_mxq_path="/tmp/vision.mxq",
    )

    assert tps_cli._effective_mxq_path_for_defaults(args) == "/tmp/global.mxq"


def test_effective_mxq_path_returns_none_when_all_absent() -> None:
    """No MXQ path should yield ``None`` so the resolver keeps its non-mobilint branch."""
    args = _base_args()

    assert tps_cli._effective_mxq_path_for_defaults(args) is None


def test_effective_mxq_path_tolerates_missing_attrs() -> None:
    """Older Namespaces without every MXQ attribute must not raise AttributeError."""
    ns = argparse.Namespace(model=_NON_MOBILINT_MODEL, vision_mxq_path="/tmp/vision.mxq")

    assert tps_cli._effective_mxq_path_for_defaults(ns) == "/tmp/vision.mxq"


def test_mobilint_model_id_still_wins_without_any_mxq_path() -> None:
    """A ``mobilint/`` model id keeps the CPU/NPU defaults independently of this change."""
    args = _base_args(model="mobilint/some-model")

    tps_cli._normalize_runtime_defaults(args)

    assert args.device == "cpu"
    assert args.device_backend == "npu"
