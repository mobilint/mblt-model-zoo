"""Tests for the face-detection vision pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch

from mblt_model_zoo.vision import MBLT_Engine, YOLO11m_face, list_models
from mblt_model_zoo.vision.face_detection import YOLO11m_face as FaceDetectionYOLO11m_face
from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.base import YOLOPostBase
from mblt_model_zoo.vision.utils.postprocess.yolo_anchor_post import YOLOAnchorPost
from mblt_model_zoo.vision.utils.postprocess.yolo_anchorless_post import YOLOAnchorlessPost
from mblt_model_zoo.vision.utils.postprocess.yolo_dflfree_post import YOLODFLFreePost
from mblt_model_zoo.vision.utils.postprocess.yolo_nmsfree_post import YOLONMSFreePost
from mblt_model_zoo.vision.utils.results import Results
from tests.npu_backend_options import BaseNpuParams, build_vision_engine_kwargs

TEST_DIR = Path(__file__).parent
IMAGE_PATH = TEST_DIR / "rc" / "cr7.jpg"
ARTIFACT_DIR = TEST_DIR / "tmp" / "face_detection"
LOCAL_ONNX_MODEL = Path("/models/ONNX/face_detection/yolo11m-face.onnx")

EXPECTED_FACE_MODELS = {
    "YOLO11l_face",
    "YOLO11m_face",
    "YOLO11n_face",
    "YOLO11s_face",
    "YOLO12l_face",
    "YOLO12m_face",
    "YOLO12n_face",
    "YOLO12s_face",
    "YOLOv10l_face",
    "YOLOv10m_face",
    "YOLOv10n_face",
    "YOLOv10s_face",
    "YOLOv6m_face",
    "YOLOv6n_face",
    "YOLOv8l_face",
    "YOLOv8m_face",
    "YOLOv8n_face",
}


def _face_pre_cfg() -> dict[str, Any]:
    """Return the representative face-detection preprocess config."""

    return {"LetterBox": {"img_size": [640, 640]}}


def _face_post_cfg() -> dict[str, Any]:
    """Return the representative face-detection postprocess config."""

    return {
        "task": "face_detection",
        "nl": 3,
        "reg_max": 16,
        "conf_thres": 0.25,
        "iou_thres": 0.7,
    }


def _build_local_engine_kwargs(
    base_kwargs: dict[str, Any],
    *,
    model_cls: str,
    model_path: Path,
    framework: str | None = None,
) -> dict[str, Any]:
    """Build vision engine kwargs for a local face model artifact."""

    engine_kwargs = dict(build_vision_engine_kwargs(base_kwargs, model_cls=model_cls))
    engine_kwargs.pop("mxq_path", None)
    engine_kwargs["model_path"] = str(model_path)
    if framework is not None:
        engine_kwargs["framework"] = framework
    return engine_kwargs


@lru_cache(maxsize=1)
def _get_reference_face_detections() -> tuple[tuple[list[float], float], ...]:
    """Return local ONNX face detections as ``((xyxy, conf), ...)``."""

    model = MBLT_Engine(model_cls="yolo11m-face", model_path=str(LOCAL_ONNX_MODEL), framework="onnx")
    try:
        output = model(model.preprocess(str(IMAGE_PATH)))
        result = model.postprocess(output)
        detections = tuple((det[:4], det[4]) for det in result.output[0].tolist())
    finally:
        model.dispose()

    return detections


def _get_reference_face_detection() -> tuple[list[float], float]:
    """Return the top local ONNX face detection as ``(xyxy, conf)``."""

    return _get_reference_face_detections()[0]


def _make_face_decode_true_output() -> torch.Tensor:
    """Build a converted decode-true face tensor for ``conversion + filter_conversion``."""

    converted = torch.zeros((1, 3, 5), dtype=torch.float32)
    (x1, y1, x2, y2), conf = _get_reference_face_detection()
    converted[0, 0] = torch.tensor(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, conf],
        dtype=torch.float32,
    )
    converted[0, 1] = torch.tensor([46.0, 210.0, 30.0, 42.0, 0.80], dtype=torch.float32)
    converted[0, 2] = torch.tensor([520.0, 225.0, 20.0, 28.0, 0.01], dtype=torch.float32)
    return converted


def _make_dfl_logits(distance: float, reg_max: int = 16) -> torch.Tensor:
    """Return DFL logits whose expectation approximates ``distance``."""

    lower = int(distance)
    upper = min(lower + 1, reg_max - 1)
    weight_upper = max(0.0, min(1.0, distance - lower))
    weight_lower = 1.0 - weight_upper
    logits = torch.full((reg_max,), -20.0, dtype=torch.float32)
    if lower == upper:
        logits[lower] = 20.0
        return logits
    logits[lower] = float(np.log(max(weight_lower, 1e-6)))
    logits[upper] = float(np.log(max(weight_upper, 1e-6)))
    return logits


def _make_face_decode_false_heads() -> list[torch.Tensor]:
    """Build raw anchorless face heads for ``rearrange + decode + NMS``."""

    outputs: list[torch.Tensor] = []
    target_candidates = []
    for (x1, y1, x2, y2), conf in _get_reference_face_detections()[:2]:
        stride = 8.0
        center_x = ((x1 + x2) / 2) / stride
        center_y = ((y1 + y2) / 2) / stride
        x_idx = int(center_x)
        y_idx = int(center_y)
        anchor_x = x_idx + 0.5
        anchor_y = y_idx + 0.5
        target_candidates.append(
            (
                (80, y_idx, x_idx),
                (
                    anchor_x - x1 / stride,
                    anchor_y - y1 / stride,
                    x2 / stride - anchor_x,
                    y2 / stride - anchor_y,
                ),
                conf,
            )
        )
    for size in (80, 40, 20):
        det = torch.full((1, size, size, 64), -10.0, dtype=torch.float32)
        cls = torch.full((1, size, size, 1), -10.0, dtype=torch.float32)
        for target_cell, target_distances, conf in target_candidates:
            target_size, y_idx, x_idx = target_cell
            if size == target_size:
                for side, distance in enumerate(target_distances):
                    det[0, y_idx, x_idx, side * 16 : (side + 1) * 16] = _make_dfl_logits(distance)
                cls[0, y_idx, x_idx, 0] = torch.logit(torch.tensor(conf, dtype=torch.float32), eps=1e-6)
        outputs.extend([det, cls])
    return outputs


def _run_face_postprocess_artifact(
    output: torch.Tensor | list[torch.Tensor],
    *,
    artifact_name: str,
) -> None:
    """Run face postprocess and save a confirmation artifact."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = ARTIFACT_DIR / artifact_name
    pre_cfg = _face_pre_cfg()
    post_cfg = _face_post_cfg()
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    result = Results(pre_cfg, post_cfg, postprocessor(output))

    assert result.task == "face_detection"
    assert result.output is not None
    plotted = result.plot(str(IMAGE_PATH), save_path=str(save_path))

    assert plotted is not None
    assert save_path.is_file()


def _run_face_inference(
    base_npu_params: BaseNpuParams,
    *,
    model_cls: str,
    model_path: Path,
    artifact_name: str,
    framework: str | None = None,
) -> None:
    """Run local face-detection inference and save a confirmation artifact."""

    save_path = ARTIFACT_DIR / artifact_name
    model = MBLT_Engine(
        **_build_local_engine_kwargs(
            base_npu_params.base,
            model_cls=model_cls,
            model_path=model_path,
            framework=framework,
        )
    )

    try:
        input_img = model.preprocess(str(IMAGE_PATH))
        output = model(input_img)
        result = model.postprocess(output)

        assert result.task == "face_detection"
        assert result.output is not None
        plotted = result.plot(str(IMAGE_PATH), save_path=str(save_path))

        assert plotted is not None
        assert save_path.is_file()
    finally:
        model.dispose()


def test_face_detection_models_exported() -> None:
    """Expose YAML-backed face models through legacy import surfaces."""

    models = set(list_models("face_detection")["face_detection"])

    assert EXPECTED_FACE_MODELS.issubset(models)
    assert YOLO11m_face is FaceDetectionYOLO11m_face


@pytest.mark.parametrize(
    ("post_cfg", "expected_type"),
    [
        ({"task": "face_detection", "nl": 3, "reg_max": 16}, YOLOAnchorlessPost),
        ({"task": "face_detection", "nl": 3, "dflfree": True}, YOLODFLFreePost),
        ({"task": "face_detection", "nl": 3, "nmsfree": True}, YOLONMSFreePost),
        (
            {
                "task": "face_detection",
                "anchors": [
                    [10, 13, 16, 30, 33, 23],
                    [30, 61, 62, 45, 59, 119],
                    [116, 90, 156, 198, 373, 326],
                ],
            },
            YOLOAnchorPost,
        ),
    ],
)
def test_face_detection_build_postprocess_routes_detection_family(
    post_cfg: dict[str, Any],
    expected_type: type[YOLOPostBase],
) -> None:
    """Reuse YOLO detection postprocessors for face detection."""

    pre_cfg = _face_pre_cfg()
    postprocessor = build_postprocess(
        pre_cfg,
        {
            "conf_thres": 0.25,
            "iou_thres": 0.7,
            **post_cfg,
        },
    )

    assert isinstance(postprocessor, expected_type)
    assert cast(YOLOPostBase, postprocessor).nc == 1


def test_face_detection_plot_uses_face_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Render face detections without depending on COCO label names."""

    labels: list[str] = []
    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "face.jpg"
    cv2.imwrite(str(source_path), np.full((64, 64, 3), 255, dtype=np.uint8))

    original_put_text = cv2.putText

    def _record_put_text(*args: Any, **kwargs: Any) -> np.ndarray:
        labels.append(args[1])
        return original_put_text(*args, **kwargs)

    monkeypatch.setattr(cv2, "putText", _record_put_text)

    result = Results(
        {"LetterBox": {"img_size": [64, 64]}},
        {"task": "face_detection"},
        [torch.tensor([[8.0, 8.0, 32.0, 32.0, 0.95, 0.0]], dtype=torch.float32)],
    )

    plotted = result.plot(str(source_path), save_path=str(save_path))

    assert plotted is not None
    assert save_path.is_file()
    assert labels == ["face 95%"]


def test_face_detection_onnx_inference_saves_artifact(base_npu_params: BaseNpuParams) -> None:
    """Run face-detection ONNX inference from a local model path."""

    assert LOCAL_ONNX_MODEL.is_file()
    _run_face_inference(
        base_npu_params,
        model_cls="yolo11m-face",
        model_path=LOCAL_ONNX_MODEL,
        artifact_name="onnx_yolo11m_face.jpg",
        framework="onnx",
    )


def test_face_detection_decode_true_postprocess_saves_artifact() -> None:
    """Run the face decode-true path and save a confirmation artifact."""

    _run_face_postprocess_artifact(
        _make_face_decode_true_output(),
        artifact_name="mxq_decode_true_yolo11m_face.jpg",
    )


def test_face_detection_decode_false_postprocess_saves_artifact() -> None:
    """Run the face decode-false path and save a confirmation artifact."""

    _run_face_postprocess_artifact(
        _make_face_decode_false_heads(),
        artifact_name="mxq_decode_false_yolo11m_face.jpg",
    )


def test_face_detection_decode_false_postprocess_returns_multiple_faces() -> None:
    """Decode-false face raw heads should keep multiple seeded detections."""

    result = cast(YOLOPostBase, build_postprocess(_face_pre_cfg(), _face_post_cfg()))(_make_face_decode_false_heads())

    assert len(result) == 1
    assert result[0].shape[0] >= 2
    assert torch.all(result[0][:2, 4] > 0.5)


def test_face_detection_non_e2e_converted_output_shape() -> None:
    """Face detection should support ``e2e=False`` converted outputs."""

    postprocessor = build_postprocess(
        _face_pre_cfg(),
        {
            **_face_post_cfg(),
            "e2e": False,
        },
    )

    result = postprocessor(_make_face_decode_true_output())

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 5, 3)


def test_face_detection_non_e2e_raw_heads_shape() -> None:
    """Face detection should support ``e2e=False`` raw-head outputs."""

    postprocessor = build_postprocess(
        _face_pre_cfg(),
        {
            **_face_post_cfg(),
            "e2e": False,
        },
    )

    result = postprocessor(_make_face_decode_false_heads())

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 1
    assert result.shape[1] == 5
    assert result.shape[2] == 8400
