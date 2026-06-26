"""Tests for vision CLI parser registration and dataset source resolution."""

from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.cli.val import _dataset_ready, _default_data_path_for_task, _resolve_coco_sources, _run_validation
from mblt_model_zoo.vision.utils.evaluation.eval_dota import _match_predictions, evaluate_dota_predictions
from mblt_model_zoo.vision.utils.evaluation.eval_widerface import WiderFaceResult, eval_widerface


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


def test_cli_predict_accepts_face_model_name() -> None:
    """Accept face-detection models through the shared predict parser."""

    parser = build_parser()
    args = parser.parse_args(["predict", "--source", "./face.jpg", "--model", "yolo11m-face"])

    assert args.model == "yolo11m-face"


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


def test_cli_val_supports_face_detection_defaults() -> None:
    """Use WiderFace as the default validation dataset for face models."""

    assert _default_data_path_for_task("face_detection").endswith(".mblt_model_zoo/datasets/widerface")


def test_cli_val_detects_organized_dotav1(tmp_path: Path) -> None:
    """Recognize an organized DOTAv1 validation layout."""

    data_path = tmp_path / "dotav1"
    (data_path / "images" / "val").mkdir(parents=True)
    (data_path / "labels" / "val").mkdir(parents=True)

    assert _dataset_ready("obb", str(data_path))


def test_cli_val_detects_organized_widerface(tmp_path: Path) -> None:
    """Recognize an organized WiderFace validation layout."""

    data_path = tmp_path / "widerface"
    (data_path / "images" / "0--Parade").mkdir(parents=True)
    for file_name in ("wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"):
        (data_path / file_name).write_bytes(b"mat")

    assert _dataset_ready("face_detection", str(data_path))


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


def test_cli_val_routes_face_detection_to_widerface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route face validation through the WiderFace evaluator."""

    data_path = tmp_path / "widerface"
    (data_path / "images" / "0--Parade").mkdir(parents=True)
    for file_name in ("wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"):
        (data_path / file_name).write_bytes(b"mat")
    calls = {}

    class _FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            calls["engine_kwargs"] = kwargs
            self.post_cfg = {"task": "face_detection"}

        def dispose(self) -> None:
            calls["disposed"] = True

    def _fake_eval_widerface(**kwargs: object) -> WiderFaceResult:
        calls["eval_kwargs"] = kwargs
        return WiderFaceResult(0.8, 0.7, 0.6)

    import mblt_model_zoo.vision as vision_module
    import mblt_model_zoo.vision.utils.evaluation as evaluation_module

    monkeypatch.setattr(vision_module, "MBLT_Engine", _FakeEngine)
    monkeypatch.setattr(evaluation_module, "eval_widerface", _fake_eval_widerface)

    args = Namespace(
        model="yolo11m-face",
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
        batch_size=4,
        conf_thres=0.3,
        iou_thres=0.6,
        force_organize=False,
        image_dir=None,
        xml_dir=None,
        annotation_dir=None,
    )

    score = _run_validation(args)

    assert score == pytest.approx((0.8 + 0.7 + 0.6) / 3)
    assert calls["engine_kwargs"]["model_cls"] == "yolo11m-face"
    assert calls["eval_kwargs"]["data_path"] == str(data_path)
    assert calls["eval_kwargs"]["batch_size"] == 4
    assert calls["eval_kwargs"]["conf_thres"] == 0.3
    assert calls["eval_kwargs"]["iou_thres"] == 0.6
    assert calls["disposed"] is True


def test_eval_widerface_formats_predictions_and_returns_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Format WiderFace predictions by event/image and return AP metrics."""

    data_path = tmp_path / "widerface"
    (data_path / "images").mkdir(parents=True)
    captured: dict[str, object] = {}

    class _FakeDataset:
        def __init__(self, root: str) -> None:
            assert root.endswith("images")
            self.samples = [
                ("image0.jpg", "0--Parade", "0_Parade_marchingband_1_5.jpg"),
                ("image1.jpg", "1--Handshaking", "1_Handshaking_Handshaking_1_10.jpg"),
            ]

        def __len__(self) -> int:
            return len(self.samples)

    def _fake_loader(
        dataset: object, batch_size: int, preprocess_fn: object
    ) -> list[tuple[object, np.ndarray, list[None], tuple[str, ...], tuple[str, ...]]]:
        del dataset, batch_size, preprocess_fn
        return [
            (
                np.zeros((2, 640, 640, 3), dtype=np.float32),
                np.array([[980, 652], [980, 652]], dtype=np.int64),
                [None, None],
                ("0--Parade", "1--Handshaking"),
                ("0_Parade_marchingband_1_5.jpg", "1_Handshaking_Handshaking_1_10.jpg"),
            )
        ]

    def _fake_norm_score(pred: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        captured["pred"] = pred
        return pred

    def _fake_evaluation(pred: dict[str, dict[str, np.ndarray]], gt_path: str, iou_thresh: float = 0.5) -> list[float]:
        captured["gt_path"] = gt_path
        captured["iou_thresh"] = iou_thresh
        captured["eval_pred"] = pred
        return [0.9, 0.8, 0.7]

    class _FakePostprocessor:
        def nmsout2eval(
            self,
            nms_output: list[torch.Tensor],
            input_shape: tuple[int, int],
            img0_shapes: list[tuple[int, int]],
            ratio_pad: list[None],
        ) -> tuple[list[list[int]], list[list[list[float]]], list[list[float]]]:
            captured["input_shape"] = input_shape
            captured["img0_shapes"] = img0_shapes
            captured["ratio_pad"] = ratio_pad
            del nms_output
            return (
                [[0], []],
                [[[10.0, 20.0, 30.0, 40.0]], []],
                [[0.95], []],
            )

    class _FakeModel:
        def __init__(self) -> None:
            self.post_cfg = {"task": "face_detection"}
            self.postprocessor = _FakePostprocessor()
            self.threshold_calls: list[tuple[float | None, float | None]] = []

        def preprocess_with_metadata(self, image: object) -> tuple[object, dict[str, object]]:
            return image, {"ratio_pad": None}

        def set_postprocess_thresholds(self, conf_thres: float | None = None, iou_thres: float | None = None) -> None:
            self.threshold_calls.append((conf_thres, iou_thres))

        def __call__(self, input_npu: object) -> object:
            return input_npu

        def postprocess(self, output: object) -> Namespace:
            del output
            return Namespace(
                output=[torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.95, 0.0]], dtype=torch.float32), torch.zeros((0, 6))]
            )

    eval_module = importlib.import_module("mblt_model_zoo.vision.utils.evaluation.eval_widerface")

    monkeypatch.setattr(eval_module, "CustomWiderface", _FakeDataset)
    monkeypatch.setattr(eval_module, "get_widerface_loader", _fake_loader)
    monkeypatch.setattr(eval_module, "norm_score", _fake_norm_score)
    monkeypatch.setattr(eval_module, "evaluation", _fake_evaluation)

    model = _FakeModel()
    result = eval_widerface(model=model, data_path=str(data_path), batch_size=2, conf_thres=0.2, iou_thres=0.4)

    assert result == WiderFaceResult(0.9, 0.8, 0.7)
    assert result.mean_ap == pytest.approx(0.8)
    assert model.threshold_calls == [(0.2, 0.4)]
    assert captured["gt_path"] == str(data_path)
    pred = captured["pred"]
    assert isinstance(pred, dict)
    assert pred["0--Parade"]["0_Parade_marchingband_1_5"].shape == (1, 5)
    assert pred["1--Handshaking"]["1_Handshaking_Handshaking_1_10"].shape == (0, 5)


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
