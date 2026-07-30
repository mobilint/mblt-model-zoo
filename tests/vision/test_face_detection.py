"""CPU regression tests for face-detection postprocessing and exports."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch

from mblt_model_zoo.vision import YOLO11m_face, list_models
from mblt_model_zoo.vision.face_detection import YOLO11m_face as FaceDetectionYOLO11mFace
from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.base import YOLODetectionPostBase
from mblt_model_zoo.vision.utils.postprocess.yolo_anchor_post import YOLOAnchorDetectionPost
from mblt_model_zoo.vision.utils.postprocess.yolo_anchorless_post import YOLOAnchorlessDetectionPost
from mblt_model_zoo.vision.utils.postprocess.yolo_dflfree_post import YOLODFLFreeDetectionPost
from mblt_model_zoo.vision.utils.postprocess.yolo_nmsfree_post import YOLONMSFreeDetectionPost
from mblt_model_zoo.vision.utils.results import Results


def _pre_cfg() -> dict[str, Any]:
    """Return a representative face preprocessing configuration."""

    return {"LetterBox": {"img_size": [640, 640]}}


def _post_cfg(**overrides: Any) -> dict[str, Any]:
    """Return a representative face postprocessing configuration."""

    return {"task": "face_detection", "nl": 3, "reg_max": 16, "conf_thres": 0.25, **overrides}


@pytest.mark.parametrize(
    ("post_cfg", "expected_type"),
    [
        ({"nl": 3, "reg_max": 16}, YOLOAnchorlessDetectionPost),
        ({"nl": 3, "dflfree": True}, YOLODFLFreeDetectionPost),
        ({"nl": 3, "nmsfree": True}, YOLONMSFreeDetectionPost),
        ({"anchors": [[10, 13, 16, 30, 33, 23]]}, YOLOAnchorDetectionPost),
    ],
)
def test_face_detection_routes_postprocessors(
    post_cfg: dict[str, Any], expected_type: type[YOLODetectionPostBase]
) -> None:
    """Route every supported face head family to its YOLO postprocessor."""

    postprocessor = build_postprocess(_pre_cfg(), _post_cfg(**post_cfg))

    assert isinstance(postprocessor, expected_type)
    assert cast(YOLODetectionPostBase, postprocessor).nc == 1


def test_face_detection_exports_and_plot_label(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Preserve legacy exports and render a face-specific detection label."""

    assert "YOLO11m_face" in list_models("face_detection")["face_detection"]
    assert YOLO11m_face is FaceDetectionYOLO11mFace
    source_path = tmp_path / "source.jpg"
    output_path = tmp_path / "face.jpg"
    cv2.imwrite(str(source_path), np.full((32, 32, 3), 255, dtype=np.uint8))
    labels: list[str] = []
    original_put_text = cv2.putText
    monkeypatch.setattr(
        cv2, "putText", lambda *args, **kwargs: (labels.append(args[1]), original_put_text(*args, **kwargs))[1]
    )

    Results(_pre_cfg(), {"task": "face_detection"}, [torch.tensor([[1, 1, 16, 16, 0.95, 0]])]).plot(
        str(source_path), save_path=str(output_path)
    )

    assert output_path.is_file()
    assert labels == ["face 95%"]


def test_face_detection_non_e2e_converted_and_raw_outputs() -> None:
    """Keep converted and raw face heads available through the legacy non-E2E contract."""

    postprocessor = build_postprocess(_pre_cfg(), _post_cfg(e2e=False))
    converted = torch.tensor(
        [[[10.0, 12.0, 0.9], [20.0, 18.0, 0.1], [8.0, 7.0, 0.2], [9.0, 6.0, 0.3], [0.8, 0.5, 0.1]]]
    )
    converted_result = postprocessor(converted)
    raw_heads = [torch.zeros((1, size, size, 64)) for size in (80, 40, 20)]
    raw_heads = [
        tensor
        for pair in zip(raw_heads, [torch.zeros((*tensor.shape[:3], 1)) for tensor in raw_heads])
        for tensor in pair
    ]
    raw_result = postprocessor(raw_heads)

    assert isinstance(converted_result, torch.Tensor)
    assert converted_result.shape == (1, 5, 3)
    assert isinstance(raw_result, torch.Tensor)
    assert raw_result.shape == (1, 5, 8400)
