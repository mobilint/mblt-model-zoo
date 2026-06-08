"""Tests for ONNX inference across representative YOLO postprocess families."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import MBLT_Engine

TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    ("model_cls", "task"),
    [
        ("yolov5m", "object_detection"),
        ("yolov5m-seg", "instance_segmentation"),
        ("yolo11m", "object_detection"),
        ("yolo11m-seg", "instance_segmentation"),
        ("yolo11m-pose", "pose_estimation"),
        ("yolov10m", "object_detection"),
        ("yolo26m", "object_detection"),
        ("yolo26m-seg", "instance_segmentation"),
        ("yolo26m-pose", "pose_estimation"),
    ],
)
def test_onnx_yolo_inference(model_cls: str, task: str) -> None:
    """Run ONNX inference for representative YOLO postprocess families."""

    image_path = os.path.join(TEST_DIR, "rc", "cr7.jpg")
    model = MBLT_Engine(model_cls=model_cls, framework="onnx")

    try:
        input_img = model.preprocess(image_path)
        output = model(input_img)
        result = model.postprocess(output)

        assert result is not None
        assert result.task == task
        assert result.output is not None
    finally:
        model.dispose()
