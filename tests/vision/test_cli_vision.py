"""Tests for vision CLI parser registration and dataset source resolution."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.cli.val import _dataset_ready, _default_data_path_for_task, _resolve_coco_sources, _run_validation
from mblt_model_zoo.vision.utils.evaluation.eval_dota import _match_predictions, evaluate_dota_predictions


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


def test_cli_val_supports_obb_defaults() -> None:
    """Use DOTAv1 as the default validation dataset for OBB models."""

    assert _default_data_path_for_task("obb").endswith(".mblt_model_zoo/datasets/dotav1")


def test_cli_val_detects_organized_dotav1(tmp_path: Path) -> None:
    """Recognize an organized DOTAv1 validation layout."""

    data_path = tmp_path / "dotav1"
    (data_path / "images" / "val").mkdir(parents=True)
    (data_path / "labels" / "val").mkdir(parents=True)

    assert _dataset_ready("obb", str(data_path))


def test_cli_val_routes_obb_to_dota_evaluator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route OBB validation through the DOTAv1 evaluator."""

    data_path = tmp_path / "dotav1"
    (data_path / "images" / "val").mkdir(parents=True)
    (data_path / "labels" / "val").mkdir(parents=True)
    calls = {}

    class _FakeEngine:
        """Minimal engine double for OBB validation routing."""

        def __init__(self, **kwargs: object) -> None:
            calls["engine_kwargs"] = kwargs
            self.post_cfg = {"task": "obb"}

        def dispose(self) -> None:
            calls["disposed"] = True

    class _FakeDOTAResult:
        """Minimal DOTAv1 metric result."""

        map50 = 0.234
        map5095 = 0.123

    def _fake_eval_dota(**kwargs: object) -> _FakeDOTAResult:
        calls["eval_kwargs"] = kwargs
        return _FakeDOTAResult()

    import mblt_model_zoo.vision as vision_module
    import mblt_model_zoo.vision.utils.evaluation as evaluation_module

    monkeypatch.setattr(vision_module, "MBLT_Engine", _FakeEngine)
    monkeypatch.setattr(evaluation_module, "eval_dota", _fake_eval_dota)

    args = Namespace(
        model="yolov8m-obb",
        model_type="DEFAULT",
        framework="onnx",
        model_path="",
        mxq_path="",
        onnx_path="",
        dev_no=0,
        core_mode="global8",
        target_cores=None,
        target_clusters=None,
        data_path=str(data_path),
        batch_size=8,
        conf_thres=None,
        iou_thres=None,
        force_organize=False,
        image_dir=None,
        xml_dir=None,
        annotation_dir=None,
    )

    score = _run_validation(args)

    assert score == 0.123
    assert calls["engine_kwargs"]["model_cls"] == "yolov8m-obb"
    assert calls["engine_kwargs"]["framework"] == "onnx"
    assert calls["eval_kwargs"]["data_path"] == str(data_path)
    assert calls["eval_kwargs"]["batch_size"] == 8
    assert calls["disposed"] is True


def test_dota_evaluator_reports_map50_and_map5095() -> None:
    """Return both DOTAv1 mAP50 and mAP50-95 metrics."""

    ground_truths = {
        "P0001": {
            "cls": torch.tensor([0], dtype=torch.int64),
            "bboxes": torch.tensor([[50.0, 50.0, 20.0, 10.0, 0.0]], dtype=torch.float32),
        }
    }
    predictions = [
        {
            "image_id": "P0001",
            "category_id": 0,
            "score": 0.9,
            "rbox": [50.0, 50.0, 20.0, 10.0, 0.0],
        }
    ]

    result = evaluate_dota_predictions(ground_truths, predictions)

    assert result.map50 > 0.99
    assert result.map5095 > 0.99


def test_dota_matching_uses_one_to_one_duplicates() -> None:
    """Match duplicate predictions to a target only once."""

    correct = _match_predictions(
        pred_classes=torch.tensor([0, 0]),
        true_classes=torch.tensor([0]),
        iou=torch.tensor([[0.60, 0.95]]),
        iouv=torch.tensor([0.5]),
    )

    assert correct.tolist() == [[True], [False]]
