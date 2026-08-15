"""Guard: batched-MXQ TPS runs must use --core-mode single.

Mirrors the ``batch benchmark only supports --core-mode single`` enforcement
in ``benchmark/transformers/benchmark_text_generation_models.py`` and
``benchmark_image_text_to_text_models.py`` so a user running
``mblt-model-zoo tps ... --core-mode global8 --batch-size 16`` on a batched
MXQ receives the same friendly ``SystemExit`` instead of a low-level
backend error mid-launch (or silently wrong throughput).

Config resolution is stubbed via ``AutoConfig.from_pretrained`` so no real
MXQ or hardware is required.
"""

from __future__ import annotations

import argparse
import importlib

import pytest

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser


class _StubConfig:
    """Minimal stand-in for a Transformers config object with an optional text_config."""

    def __init__(self, max_batch_size=None, text_config=None, vision_config=None):
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if text_config is not None:
            self.text_config = text_config
        if vision_config is not None:
            self.vision_config = vision_config


def _stub_autoconfig(monkeypatch, config) -> None:
    """Patch ``AutoConfig.from_pretrained`` so the guard sees ``config`` without any load."""
    auto_config = importlib.import_module("transformers").AutoConfig
    monkeypatch.setattr(auto_config, "from_pretrained", lambda *a, **kw: config)


def _parse_tps_measure(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct-Batch16",
            *extra,
        ]
    )


def _parse_tps_sweep(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct-Batch16",
            "--no-plot",
            *extra,
        ]
    )


@pytest.mark.parametrize("core_mode", ["global4", "global8"])
def test_measure_rejects_non_single_core_mode_on_batched_mxq(monkeypatch, core_mode):
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--core-mode", core_mode, "--batch-size", "16")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._cmd_measure(args)

    message = str(excinfo.value)
    assert "batched MXQ only supports --core-mode single" in message
    assert "config max_batch_size=16" in message
    assert core_mode in message


def test_measure_accepts_single_core_mode_on_batched_mxq(monkeypatch):
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--core-mode", "single", "--batch-size", "16")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "single"


def test_measure_forces_single_when_core_mode_unspecified_on_batched_mxq(monkeypatch):
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--batch-size", "16")
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "single"


def test_measure_allows_non_single_core_mode_on_non_batch_mxq(monkeypatch):
    """Non-batch MXQ (config max_batch_size == 1) keeps every core_mode; sw-batch across N slots is orthogonal."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure("--core-mode", "global8", "--batch-size", "4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global8"


def test_measure_allows_non_single_core_mode_when_config_missing_max_batch_size(monkeypatch):
    """A config without ``max_batch_size`` behaves as a non-batch target — no rejection, no forcing."""
    _stub_autoconfig(monkeypatch, _StubConfig())

    args = _parse_tps_measure("--core-mode", "global4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_measure_allows_non_single_when_autoconfig_fails(monkeypatch):
    """AutoConfig failures fall through as non-batch — the pipeline path surfaces the real error."""
    auto_config = importlib.import_module("transformers").AutoConfig

    def _fail(*a, **kw):
        raise OSError("simulated network failure")

    monkeypatch.setattr(auto_config, "from_pretrained", _fail)

    args = _parse_tps_measure("--core-mode", "global4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_sweep_rejects_non_single_core_mode_on_batched_mxq(monkeypatch):
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_sweep("--core-mode", "global8", "--batch-size", "16")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._cmd_sweep(args)

    assert "batched MXQ only supports --core-mode single" in str(excinfo.value)


def test_measure_vlm_probes_text_config_max_batch_size(monkeypatch):
    """VLM releases carry ``max_batch_size`` on ``text_config``; the probe follows the same candidate order."""
    config = _StubConfig(text_config=_StubConfig(max_batch_size=4))
    _stub_autoconfig(monkeypatch, config)

    args = _parse_tps_measure("--task", "image-text-to-text", "--core-mode", "global4", "--batch-size", "4")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._cmd_measure(args)

    assert "batched MXQ only supports --core-mode single" in str(excinfo.value)


def test_probe_config_max_batch_size_uses_task_specific_vlm_vision_config(monkeypatch):
    """The probe forwards ``task`` to :func:`_candidate_max_batch_sizes`, exposing vision_config on VLM."""
    config = _StubConfig(vision_config=_StubConfig(max_batch_size=2))
    _stub_autoconfig(monkeypatch, config)

    assert (
        tps_cli._probe_config_max_batch_size(
            "mobilint/Qwen3-VL-Batch2",
            trust_remote_code=True,
            revision=None,
            task="image-text-to-text",
        )
        == 2
    )


# ---------------------------------------------------------------------------
# --mxq-path override: the guard must classify batching from the selected
# artifact's compiled ``K``, not the config's ``max_batch_size`` declaration.
# ---------------------------------------------------------------------------


def _stub_artifact_probe(monkeypatch, k_by_path: dict[str, int | None] | None = None) -> None:
    """Patch :func:`_probe_mxq_artifact_k` to return ``k`` for the given path.

    Bypasses ``qbruntime.Model`` entirely so no hardware or MXQ file is
    required. ``k_by_path=None`` (or an unknown path) returns ``None``,
    forcing the guard to fall through to the config-based probe.
    """
    mapping = dict(k_by_path or {})
    monkeypatch.setattr(tps_cli, "_probe_mxq_artifact_k", lambda path: mapping.get(path))


def test_measure_overrides_config_batch_when_mxq_path_probes_k1(monkeypatch):
    """A Batch16 release overridden with a locally-probed K=1 MXQ must not be rejected."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k1.mxq": 1})

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "/tmp/override-k1.mxq")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_measure_rejects_when_mxq_path_probes_k_gt_1_on_batch1_config(monkeypatch):
    """A batch-1 release overridden with a locally-probed K=4 MXQ must be rejected under global4."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k4.mxq": 4})

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "/tmp/override-k4.mxq")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    message = str(excinfo.value)
    assert "batched MXQ only supports --core-mode single" in message
    assert "artifact K=4" in message
    assert "global4" in message


def test_measure_falls_back_to_config_when_artifact_probe_returns_none(monkeypatch):
    """A non-local or unprobeable override falls back to the config-declared batch size."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))
    _stub_artifact_probe(monkeypatch, {})  # every path returns None

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "hf://mobilint/Some-Non-Local")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    message = str(excinfo.value)
    assert "config max_batch_size=16" in message
    assert "global4" in message


def test_measure_forces_single_when_unspecified_core_mode_on_probed_batched_artifact(monkeypatch):
    """Even without --core-mode, a probed K>1 override pins core_mode=single."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k8.mxq": 8})

    args = _parse_tps_measure("--mxq-path", "/tmp/override-k8.mxq")
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "single"


def test_measure_prefers_base_mxq_path_for_eagle3_over_bare_mxq_path(monkeypatch):
    """EAGLE-3 --base-mxq-path takes precedence: the base MXQ is the LLM that governs batching."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(
        monkeypatch,
        {"/tmp/bare-k1.mxq": 1, "/tmp/base-k4.mxq": 4},
    )

    args = _parse_tps_measure(
        "--core-mode",
        "global4",
        "--mxq-path",
        "/tmp/bare-k1.mxq",
        "--base-mxq-path",
        "/tmp/base-k4.mxq",
    )

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert "artifact K=4" in str(excinfo.value)


def test_measure_vlm_prefers_text_mxq_path_over_bare_mxq_path(monkeypatch):
    """VLM --text-mxq-path takes precedence over --mxq-path for the batched-guard check."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=1)))
    _stub_artifact_probe(
        monkeypatch,
        {"/tmp/bare.mxq": 8, "/tmp/text-k1.mxq": 1},
    )

    args = _parse_tps_measure(
        "--task",
        "image-text-to-text",
        "--core-mode",
        "global4",
        "--mxq-path",
        "/tmp/bare.mxq",
        "--text-mxq-path",
        "/tmp/text-k1.mxq",
    )

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_select_llm_mxq_override_precedence():
    """Precedence: text_mxq_path (VLM) > base_mxq_path > mxq_path; None when no override."""
    parser = build_parser()

    # No override at all.
    args = parser.parse_args(["tps", "measure", "--model", "m"])
    assert tps_cli._select_llm_mxq_override(args) is None

    # Plain --mxq-path only.
    args = parser.parse_args(["tps", "measure", "--model", "m", "--mxq-path", "/a.mxq"])
    assert tps_cli._select_llm_mxq_override(args) == "/a.mxq"

    # --base-mxq-path wins over --mxq-path for EAGLE-3-shaped releases.
    args = parser.parse_args(["tps", "measure", "--model", "m", "--mxq-path", "/a.mxq", "--base-mxq-path", "/base.mxq"])
    assert tps_cli._select_llm_mxq_override(args) == "/base.mxq"

    # VLM: --text-mxq-path wins.
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--task",
            "image-text-to-text",
            "--model",
            "m",
            "--mxq-path",
            "/a.mxq",
            "--text-mxq-path",
            "/text.mxq",
        ]
    )
    assert tps_cli._select_llm_mxq_override(args) == "/text.mxq"


def test_probe_mxq_artifact_k_returns_none_for_non_local_path(tmp_path):
    """The probe short-circuits for a non-existent path so no ``qbruntime.Model`` is instantiated."""
    missing = tmp_path / "does_not_exist.mxq"
    assert tps_cli._probe_mxq_artifact_k(str(missing)) is None
    assert tps_cli._probe_mxq_artifact_k("") is None
