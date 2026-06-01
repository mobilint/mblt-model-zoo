"""VLM routing and parser behavior tests for TPS CLI."""

from __future__ import annotations

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

    assert tps_cli._cmd_sweep(type("Args", (), {"task": "image-text-to-text"})()) == 0
    assert tps_cli._cmd_sweep(type("Args", (), {"task": "text-generation"})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]


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

    assert tps_cli._cmd_measure(type("Args", (), {"task": "image-text-to-text"})()) == 0
    assert tps_cli._cmd_measure(type("Args", (), {"task": "text-generation"})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]
