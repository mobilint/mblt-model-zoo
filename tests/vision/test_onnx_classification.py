"""Tests for vision model ONNX inference on image classification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision.image_classification import AlexNet, CAFormer_B36, YOLO26sCls

TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    "model_cls",
    [
        AlexNet,
        CAFormer_B36,
        YOLO26sCls,
    ],
)
def test_onnx_classification(model_cls) -> None:
    """Run ONNX inference for representative classification models."""
    image_path = os.path.join(TEST_DIR, "rc", "volcano.jpg")

    model = model_cls(framework="onnx")

    try:
        input_img = model.preprocess(image_path)
        output = model(input_img)
        result = model.postprocess(output)

        assert result is not None
        assert result.task == "image_classification"
        assert result.acc is not None
        assert result.output is not None
    finally:
        model.dispose()
