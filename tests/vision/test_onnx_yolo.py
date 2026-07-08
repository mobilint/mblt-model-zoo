"""Tests for ONNX inference across representative YOLO postprocess families."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import MBLT_Engine

TEST_DIR = Path(__file__).parent
TMP_DIR = TEST_DIR / "tmp"


@pytest.mark.parametrize(
    ("model_cls", "task", "image_name"),
    [
        ("yolov5m", "object_detection", "cr7.jpg"),
        ("yolov5m-seg", "instance_segmentation", "cr7.jpg"),
        ("yolo11m", "object_detection", "cr7.jpg"),
        ("yolo11m-seg", "instance_segmentation", "cr7.jpg"),
        ("yolo11m-pose", "pose_estimation", "cr7.jpg"),
        ("yolov10m", "object_detection", "cr7.jpg"),
        ("yolo26m", "object_detection", "cr7.jpg"),
        ("yolo26m-seg", "instance_segmentation", "cr7.jpg"),
        ("yolo26m-pose", "pose_estimation", "cr7.jpg"),
        ("yolov8m-obb", "obb", "airport.jpg"),
        ("yolo11m-obb", "obb", "airport.jpg"),
        ("yolo26m-obb", "obb", "airport.jpg"),
    ],
)
def test_onnx_yolo_inference(model_cls: str, task: str, image_name: str) -> None:
    """Run ONNX inference for representative YOLO postprocess families."""

    image_path = os.path.join(TEST_DIR, "rc", image_name)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    save_path = TMP_DIR / f"{model_cls}_visualization.jpg"
    model = MBLT_Engine(model_cls=model_cls, framework="onnx")

    try:
        input_img = model.preprocess(image_path)
        output = model(input_img)
        result = model.postprocess(output)

        assert result is not None
        assert result.task == task
        assert result.output is not None
        result.plot(image_path, save_path=str(save_path))
        assert save_path.is_file()
    finally:
        model.dispose()
