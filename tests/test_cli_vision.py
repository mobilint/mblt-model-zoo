"""Tests for vision CLI parser registration."""

from __future__ import annotations

import pytest

from mblt_model_zoo.cli.main import build_parser


def test_cli_predict_example_parses() -> None:
    """Parse the unified vision prediction command shape."""

    parser = build_parser()
    args = parser.parse_args(["predict", "--source", "./cat.png", "--model", "resnet50"])

    assert args.source == "./cat.png"
    assert args.model == "resnet50"
    assert args.core_mode == "global8"
    assert args.topk == 5
    assert args.conf_thres == 0.25
    assert args.iou_thres is None


def test_cli_val_defaults_to_model_thresholds() -> None:
    """Leave validation thresholds unset so model YAML defaults are used."""

    parser = build_parser()
    args = parser.parse_args(["val", "--model", "yolo11m"])

    assert args.core_mode == "global8"
    assert args.conf_thres is None
    assert args.iou_thres is None


@pytest.mark.parametrize(
    "command",
    [
        "predict",
        "classify",
        "detect",
        "pose",
        "segment",
    ],
)
def test_cli_vision_commands_parse_thresholds(command: str) -> None:
    """Parse unified vision command and compatibility aliases."""

    parser = build_parser()
    args = parser.parse_args(
        [
            command,
            "--source",
            "./image.jpg",
            "--model",
            "yolo11m",
            "--conf-thres",
            "0.5",
            "--iou-thres",
            "0.6",
        ]
    )

    assert args.conf_thres == 0.5
    assert args.iou_thres == 0.6
