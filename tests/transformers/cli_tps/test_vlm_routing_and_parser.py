"""VLM routing and parser behavior tests for TPS CLI."""

from __future__ import annotations

import argparse

import pytest

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser


def test_cli_tps_sweep_vlm_options_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Qwen2-VL-2B-Instruct",
            "--task",
            "image-text-to-text",
            "--prefill-range",
            "512:2048:512",
            "--cache-lengths",
            "128,512,1024,2048",
            "--decode-window",
            "32",
            "--image-resolutions",
            "224,448",
            "--llm-resolution",
            "224",
            "--prompt",
            "Describe.",
            "--no-plot",
        ]
    )

    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32
    assert args.image_resolutions == [224, 448]
    assert args.llm_resolution == 224
    assert args.prompt == "Describe."
    assert args.plot is None


def test_cli_tps_vlm_sweep_removed():
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tps", "vlm-sweep", "--model", "dummy"])

    assert excinfo.value.code == 2


def test_cmd_sweep_routes_vlm_task(monkeypatch):
    calls: list[str] = []

    def fake_vlm_sweep(args):
        calls.append(f"vlm:{args.task}")
        return 0

    def fake_text_sweep(args):
        calls.append(f"text:{args.task}")
        return 0

    monkeypatch.setattr(tps_cli, "_run_vlm_sweep", fake_vlm_sweep)
    monkeypatch.setattr(tps_cli, "_run_text_sweep", fake_text_sweep)

    assert tps_cli._cmd_sweep(type("Args", (), {"task": "image-text-to-text", "task_explicit": True})()) == 0
    assert tps_cli._cmd_sweep(type("Args", (), {"task": "text-generation", "task_explicit": True})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]


class _SentinelPipelineBuilt(Exception):
    """Raised by the stubbed ``_build_pipeline`` to short-circuit VLM measure setup."""


def _make_vlm_measure_args(**overrides) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "task": "image-text-to-text",
        "model": "dummy",
        "tokenizer": None,
        "device": None,
        "trust_remote_code": True,
        "dtype": None,
        "device_map": None,
        "revision": None,
        "embedding_weight": None,
        "mxq_path": None,
        "device_backend": None,
        "core_mode": "single",
        "target_cores": None,
        "target_clusters": None,
        "batch_size": None,
        "print_output": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_vlm_measure_warns_when_print_output_set(monkeypatch):
    """VLM measure warns and clears ``print_output`` before invoking pipeline construction."""

    def _stop_after_guard(_args):
        raise _SentinelPipelineBuilt()

    monkeypatch.setattr(tps_cli, "_normalize_runtime_defaults", lambda args: None)
    # ``_extract_eagle3_pipeline_kwargs`` is the first call after the print-output
    # guard; short-circuit here so the test does not need full CLI namespace.
    monkeypatch.setattr(tps_cli, "_extract_eagle3_pipeline_kwargs", _stop_after_guard)

    args = _make_vlm_measure_args(print_output=True)
    with pytest.warns(UserWarning, match="--print-output is only supported for text-only"):
        with pytest.raises(_SentinelPipelineBuilt):
            tps_cli._run_vlm_measure(args)

    assert args.print_output is False


def test_run_vlm_measure_no_warning_when_print_output_absent(monkeypatch):
    """VLM measure emits no ``--print-output`` warning when the flag is unset."""

    def _stop_after_guard(_args):
        raise _SentinelPipelineBuilt()

    monkeypatch.setattr(tps_cli, "_normalize_runtime_defaults", lambda args: None)
    monkeypatch.setattr(tps_cli, "_extract_eagle3_pipeline_kwargs", _stop_after_guard)

    args = _make_vlm_measure_args(print_output=False)
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(_SentinelPipelineBuilt):
            tps_cli._run_vlm_measure(args)

    print_output_warnings = [w for w in record if "--print-output" in str(w.message)]
    assert print_output_warnings == []


def test_cli_tps_measure_print_output_help_marks_text_only():
    parser = build_parser()
    formatter = parser.format_help  # noqa: F841 - unused

    # Format the help for the measure subparser.
    tps_sub = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            tps_sub = action.choices.get("tps")
            break
    assert tps_sub is not None
    measure_sub = None
    for action in tps_sub._actions:
        if isinstance(action, argparse._SubParsersAction):
            measure_sub = action.choices.get("measure")
            break
    assert measure_sub is not None
    help_text = measure_sub.format_help()
    assert "text-only" in help_text
    assert "ignored for VLM" in help_text


def test_cmd_measure_routes_vlm_task(monkeypatch):
    calls: list[str] = []

    def fake_vlm_measure(args):
        calls.append(f"vlm:{args.task}")
        return 0

    def fake_text_measure(args):
        calls.append(f"text:{args.task}")
        return 0

    monkeypatch.setattr(tps_cli, "_run_vlm_measure", fake_vlm_measure)
    monkeypatch.setattr(tps_cli, "_run_text_measure", fake_text_measure)

    assert tps_cli._cmd_measure(type("Args", (), {"task": "image-text-to-text", "task_explicit": True})()) == 0
    assert tps_cli._cmd_measure(type("Args", (), {"task": "text-generation", "task_explicit": True})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]
