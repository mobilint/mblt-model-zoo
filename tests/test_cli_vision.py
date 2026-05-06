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
    assert args.iou_thres == 0.45


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
