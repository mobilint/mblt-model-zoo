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
