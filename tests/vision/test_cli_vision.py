"""Tests for vision CLI parser registration and dataset source resolution."""

from __future__ import annotations

import importlib
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pytest
import torch

from mblt_model_zoo.cli._vision import run_vision_inference
from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.cli.val import _dataset_ready, _default_data_path_for_task, _resolve_coco_sources, _run_validation
from mblt_model_zoo.vision.datasets import (
    get_dataset_category_ids,
    get_dataset_class_names,
    get_dataset_config,
    get_dataset_config_for_task,
)
from mblt_model_zoo.vision.utils.datasets import get_coco_inv, get_coco_label, get_dotav1_label, get_imagenet_label
from mblt_model_zoo.vision.utils.datasets.dataloader import CustomDOTAv1
from mblt_model_zoo.vision.utils.datasets.organizer import construct_dotav1_from_archives
from mblt_model_zoo.vision.utils.evaluation.eval_dota import (
    _load_ground_truths,
    _match_predictions,
    evaluate_dota_predictions,
)
from mblt_model_zoo.vision.utils.evaluation.eval_widerface import WiderFaceResult, eval_widerface

if TYPE_CHECKING:
    from mblt_model_zoo.vision.wrapper import MBLT_Engine


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
    assert args.e2e is None


def test_vision_dataset_yaml_registry_resolves_dotav1() -> None:
    """Load DOTAv1 defaults and task routing from the YAML dataset registry."""

    config = get_dataset_config("dotav1")

    assert config["val"] == "images/val"
    assert config["download"]["type"] == "google_drive_folder"
    assert get_dataset_config_for_task("obb")["name"] == config["name"]


def test_vision_dataset_yaml_registry_preserves_class_metadata() -> None:
    """Serve legacy class helper values from the YAML dataset definitions."""

    assert get_dataset_class_names("coco")[0] == "person"
    assert get_dataset_category_ids("coco") is get_dataset_category_ids("coco")
    assert get_coco_label(79) == "toothbrush"
    assert get_coco_inv(11) == 13
    assert get_dotav1_label(2) == "storage-tank"
    assert get_imagenet_label(0) == "tench"


@pytest.mark.parametrize(
    ("command", "option", "expected"),
    [
        ("predict", "--e2e", True),
        ("predict", "--e2e false", False),
        ("val", "--e2e", True),
        ("val", "--e2e false", False),
    ],
)
def test_cli_vision_commands_parse_e2e_option(command: str, option: str, expected: bool) -> None:
    """Parse the optional YOLO end-to-end postprocessing override."""

    parser = build_parser()
    command_args = [command, "--model", "yolo11m", *option.split()]
    if command == "predict":
        command_args.extend(["--source", "./cat.png"])

    args = parser.parse_args(command_args)

    assert args.e2e is expected


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


def test_cli_predict_applies_face_detection_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forward predict thresholds to the face-detection postprocessor."""

    source_path = tmp_path / "face.jpg"
    source_path.write_bytes(b"fake")
    calls: dict[str, object] = {}

    class _FakeResult:
        def plot(self, source_path: str, save_path: str, **kwargs: object) -> None:
            calls["plot"] = {
                "source_path": source_path,
                "save_path": save_path,
                "kwargs": kwargs,
            }

    class _FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            calls["engine_kwargs"] = kwargs
            self.post_cfg = {"task": "face_detection"}

        def set_postprocess_thresholds(self, conf_thres: float | None = None, iou_thres: float | None = None) -> None:
            calls["thresholds"] = (conf_thres, iou_thres)

        def preprocess(self, source: str) -> str:
            calls["preprocess"] = source
            return "preprocessed"

        def __call__(self, input_img: str) -> str:
            calls["forward"] = input_img
            return "raw-output"

        def postprocess(self, output: str, **kwargs: object) -> _FakeResult:
            calls["postprocess"] = {"output": output, "kwargs": kwargs}
            return _FakeResult()

        def dispose(self) -> None:
            calls["disposed"] = True

    import mblt_model_zoo.cli._vision as vision_cli_module

    monkeypatch.setattr(vision_cli_module, "require_source_file", lambda source: None)
    monkeypatch.setattr(
        vision_cli_module,
        "resolve_output_path",
        lambda output, command, source, model: str(tmp_path / "out.jpg"),
    )
    monkeypatch.setattr("mblt_model_zoo.vision.MBLT_Engine", _FakeEngine)

    args = Namespace(
        source=str(source_path),
        model="yolo11m-face",
        output="",
        framework="onnx",
        model_path="/models/ONNX/face_detection/yolo11m-face.onnx",
        mxq_path="",
        onnx_path="",
        model_type="DEFAULT",
        core_mode="global8",
        dev_no=0,
        target_cores=None,
        target_clusters=None,
        topk=5,
        conf_thres=0.25,
        iou_thres=0.6,
        e2e=True,
    )

    run_vision_inference(args, command="predict")

    engine_kwargs = cast(dict[str, object], calls["engine_kwargs"])
    assert calls["thresholds"] == (0.25, 0.6)
    assert engine_kwargs["postprocess_kwargs"] == {"e2e": True}
    assert calls["disposed"] is True


def test_cli_predict_applies_obb_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forward predict thresholds to the OBB postprocessor."""

    source_path = tmp_path / "airport.jpg"
    source_path.write_bytes(b"fake")
    calls: dict[str, object] = {}

    class _FakeResult:
        def plot(self, source_path: str, save_path: str, **kwargs: object) -> None:
            calls["plot"] = {
                "source_path": source_path,
                "save_path": save_path,
                "kwargs": kwargs,
            }

    class _FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            calls["engine_kwargs"] = kwargs
            self.post_cfg = {"task": "obb"}

        def set_postprocess_thresholds(self, conf_thres: float | None = None, iou_thres: float | None = None) -> None:
            calls["thresholds"] = (conf_thres, iou_thres)

        def preprocess(self, source: str) -> str:
            calls["preprocess"] = source
            return "preprocessed"

        def __call__(self, input_img: str) -> str:
            calls["forward"] = input_img
            return "raw-output"

        def postprocess(self, output: str, **kwargs: object) -> _FakeResult:
            calls["postprocess"] = {"output": output, "kwargs": kwargs}
            return _FakeResult()

        def dispose(self) -> None:
            calls["disposed"] = True

    import mblt_model_zoo.cli._vision as vision_cli_module

    monkeypatch.setattr(vision_cli_module, "require_source_file", lambda source: None)
    monkeypatch.setattr(
        vision_cli_module,
        "resolve_output_path",
        lambda output, command, source, model: str(tmp_path / "out.jpg"),
    )
    monkeypatch.setattr("mblt_model_zoo.vision.MBLT_Engine", _FakeEngine)

    args = Namespace(
        source=str(source_path),
        model="yolov8m-obb",
        output="",
        framework="onnx",
        model_path="/models/ONNX/obb/yolov8m-obb.onnx",
        mxq_path="",
        onnx_path="",
        model_type="DEFAULT",
        core_mode="global8",
        dev_no=0,
        target_cores=None,
        target_clusters=None,
        topk=5,
        conf_thres=0.25,
        iou_thres=0.6,
    )

    run_vision_inference(args, command="predict")

    assert calls["thresholds"] == (0.25, 0.6)
    assert calls["disposed"] is True


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


def test_cli_val_detects_original_dotav1_labels(tmp_path: Path) -> None:
    """Recognize DOTAv1 validation layouts that keep original labels."""

    data_path = tmp_path / "dotav1"
    (data_path / "images" / "val").mkdir(parents=True)
    (data_path / "labels" / "val_original").mkdir(parents=True)

    assert _dataset_ready("obb", str(data_path))


def test_google_drive_dotav1_archives_preserve_legacy_evaluation_layout(tmp_path: Path) -> None:
    """Organize Drive archives into the original-label layout used by DOTAv1 evaluation."""

    image_source = tmp_path / "P0001.png"
    assert cv2.imwrite(str(image_source), np.zeros((16, 20, 3), dtype=np.uint8))
    label_source = tmp_path / "P0001.txt"
    label_source.write_text(
        "0 0 10 0 10 10 0 10 plane 0\n0 1 10 1 10 11 0 11 ship 2\n",
        encoding="utf-8",
    )
    image_archive = tmp_path / "part1.zip"
    label_archive = tmp_path / "labelTxt.zip"
    with zipfile.ZipFile(image_archive, "w") as archive:
        archive.write(image_source, "images/P0001.png")
    with zipfile.ZipFile(label_archive, "w") as archive:
        archive.write(label_source, "labelTxt/P0001.txt")

    output_dir = tmp_path / "dotav1"
    stale_label_dir = output_dir / "labels" / "val"
    stale_label_dir.mkdir(parents=True)
    (stale_label_dir / "P0001.txt").write_text("0 0 0 0 0 0 0 0 plane 0\n", encoding="utf-8")
    stale_image_path = output_dir / "images" / "val" / "stale.png"
    stale_image_path.parent.mkdir(parents=True)
    stale_image_path.write_bytes(b"stale")
    construct_dotav1_from_archives(str(image_archive), str(label_archive), str(output_dir))

    assert (output_dir / "images" / "val" / "P0001.png").is_file()
    assert not stale_image_path.exists()
    assert not stale_label_dir.exists()
    assert (output_dir / "labels" / "val_original" / "P0001.txt").is_file()
    ground_truths = _load_ground_truths(str(output_dir), CustomDOTAv1(str(output_dir)))
    assert ground_truths["P0001"]["cls"].tolist() == [0]


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
            self.postprocessor = Namespace(e2e=True)

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
        e2e=True,
        force_organize=False,
        image_dir=None,
        xml_dir=None,
        annotation_dir=None,
    )

    score = _run_validation(args)

    engine_kwargs = cast(dict[str, object], calls["engine_kwargs"])
    eval_kwargs = cast(dict[str, object], calls["eval_kwargs"])
    assert score == 0.123
    assert engine_kwargs["model_cls"] == "yolov8m-obb"
    assert engine_kwargs["framework"] == "onnx"
    assert engine_kwargs["postprocess_kwargs"] == {"e2e": True}
    assert eval_kwargs["data_path"] == str(data_path)
    assert eval_kwargs["batch_size"] == 8
    assert calls["disposed"] is True


def test_cli_val_rejects_non_e2e_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject export-style YOLO outputs, which validation cannot evaluate."""

    class _FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            self.post_cfg = {"task": "object_detection"}
            self.postprocessor = Namespace(e2e=False)

        def dispose(self) -> None:
            pass

    import mblt_model_zoo.vision as vision_module

    monkeypatch.setattr(vision_module, "MBLT_Engine", _FakeEngine)
    args = Namespace(
        model="yolo11m",
        model_type="DEFAULT",
        framework=None,
        model_path="",
        mxq_path="",
        onnx_path="",
        dev_no=0,
        core_mode="global8",
        target_cores=None,
        target_clusters=None,
        e2e=False,
    )

    with pytest.raises(SystemExit, match="Validation requires end-to-end YOLO postprocessing"):
        _run_validation(args)


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

    engine_kwargs = cast(dict[str, object], calls["engine_kwargs"])
    eval_kwargs = cast(dict[str, object], calls["eval_kwargs"])
    assert score == pytest.approx((0.8 + 0.7 + 0.6) / 3)
    assert engine_kwargs["model_cls"] == "yolo11m-face"
    assert eval_kwargs["data_path"] == str(data_path)
    assert eval_kwargs["batch_size"] == 4
    assert eval_kwargs["conf_thres"] == 0.3
    assert eval_kwargs["iou_thres"] == 0.6
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
    result = eval_widerface(
        model=cast("MBLT_Engine", model),
        data_path=str(data_path),
        batch_size=2,
        conf_thres=0.2,
        iou_thres=0.4,
    )

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

    assert correct.tolist() == [[False], [True]]


def test_dota_ground_truth_loader_skips_difficult_original_labels(tmp_path: Path) -> None:
    """Ignore difficult objects from original-layout DOTAv1 labels."""

    data_path = tmp_path / "dotav1"
    label_dir = data_path / "labels" / "val_original"
    label_dir.mkdir(parents=True)
    (label_dir / "P0001.txt").write_text(
        "\n".join(
            [
                "10 10 20 10 20 20 10 20 plane 0",
                "30 30 40 30 40 40 30 40 ship 1",
                "45 45 55 45 55 55 45 55 storage-tank 2",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeDataset:
        ids = ["P0001"]
        image_paths = ["P0001.png"]

        def _load_image(self, image_path: str) -> np.ndarray:
            assert image_path == "P0001.png"
            return np.zeros((64, 64, 3), dtype=np.uint8)

    ground_truths = _load_ground_truths(str(data_path), cast("CustomDOTAv1", _FakeDataset()))

    assert ground_truths["P0001"]["cls"].tolist() == [0]
    assert ground_truths["P0001"]["polygons"].shape == (1, 4, 2)
    assert ground_truths["P0001"]["bboxes"].shape == (1, 5)
