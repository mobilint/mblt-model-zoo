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
    assert "max_batch_size=16" in message
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
