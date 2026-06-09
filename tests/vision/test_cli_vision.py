"""Tests for vision CLI parser registration and dataset source resolution."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.cli.val import _resolve_coco_sources


def test_cli_predict_example_parses() -> None:
    """Parse the unified vision prediction command shape."""

    parser = build_parser()
    args = parser.parse_args(["predict", "--source", "./cat.png", "--model", "resnet50"])

    assert args.source == "./cat.png"
    assert args.model == "resnet50"
    assert args.framework is None
    assert args.model_path == ""
    assert args.mxq_path == ""
    assert args.onnx_path == ""
    assert args.core_mode == "global8"
    assert args.topk == 5
    assert args.conf_thres == 0.25
    assert args.iou_thres is None


def test_cli_val_defaults_to_model_thresholds() -> None:
    """Leave validation thresholds unset so model YAML defaults are used."""

    parser = build_parser()
    args = parser.parse_args(["val", "--model", "yolo11m"])

    assert args.framework is None
    assert args.model_path == ""
    assert args.mxq_path == ""
    assert args.onnx_path == ""
    assert args.core_mode == "global8"
    assert args.conf_thres is None
    assert args.iou_thres is None


def test_cli_predict_parses_onnx_framework_and_model_path() -> None:
    """Accept the shared ONNX framework options for prediction."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--source",
            "./cat.png",
            "--model",
            "alexnet",
            "--framework",
            "onnx",
            "--model-path",
            "./alexnet.onnx",
        ]
    )

    assert args.framework == "onnx"
    assert args.model_path == "./alexnet.onnx"


def test_cli_predict_parses_mxq_model_path() -> None:
    """Accept the shared local MXQ path option for prediction."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--source",
            "./cat.png",
            "--model",
            "resnet50",
            "--model-path",
            "./resnet50.mxq",
        ]
    )

    assert args.framework is None
    assert args.model_path == "./resnet50.mxq"
    assert args.mxq_path == ""
    assert args.onnx_path == ""


def test_cli_predict_preserves_framework_specific_model_paths() -> None:
    """Keep compatibility path aliases separate from the generic model path."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--source",
            "./cat.png",
            "--model",
            "resnet50",
            "--framework",
            "onnx",
            "--mxq-path",
            "./resnet50.mxq",
        ]
    )

    assert args.framework == "onnx"
    assert args.model_path == ""
    assert args.mxq_path == "./resnet50.mxq"
    assert args.onnx_path == ""


def test_cli_val_preserves_framework_specific_model_paths() -> None:
    """Keep validation compatibility path aliases separate from the generic model path."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "val",
            "--model",
            "resnet50",
            "--framework",
            "onnx",
            "--mxq-path",
            "./resnet50.mxq",
            "--onnx-path",
            "./resnet50.onnx",
        ]
    )

    assert args.framework == "onnx"
    assert args.model_path == ""
    assert args.mxq_path == "./resnet50.mxq"
    assert args.onnx_path == "./resnet50.onnx"


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


def test_cli_val_reuses_extracted_coco_annotation_parent(tmp_path: Path) -> None:
    """Resolve the common extracted COCO annotations layout without nesting twice."""

    workspace = tmp_path
    data_path = workspace / "datasets" / "coco"
    data_path.mkdir(parents=True)
    search_root = data_path.parent
    (search_root / "val2017").mkdir()
    annotations_leaf = search_root / "annotations"
    annotations_leaf.mkdir()

    args = Namespace(
        image_dir=None,
        annotation_dir=None,
        force_organize=False,
    )

    image_dir, annotation_dir = _resolve_coco_sources(args, str(data_path))

    assert image_dir == str(search_root / "val2017")
    assert annotation_dir == str(search_root)
