"""Tests for vision model MXQ inference on image classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from mblt_model_zoo.vision import MBLT_Engine
from tests.npu_backend_options import BaseNpuParams, build_vision_engine_kwargs

TEST_DIR = Path(__file__).parent


@pytest.fixture(params=["alexnet", "caformer_b36", "yolo26s-cls"])
def mxq_model(request: pytest.FixtureRequest, base_npu_params: BaseNpuParams) -> Generator[MBLT_Engine, None, None]:
    """Create representative MXQ classification engines and dispose them after use."""

    model_kwargs = build_vision_engine_kwargs(base_npu_params.base, model_cls=str(request.param))
    model = MBLT_Engine(**model_kwargs)
    yield model
    model.dispose()


def test_mxq_classification(mxq_model: MBLT_Engine) -> None:
    """Run MXQ inference for representative classification models."""

    image_path = os.path.join(TEST_DIR, "rc", "volcano.jpg")

    input_img = mxq_model.preprocess(image_path)
    output = mxq_model(input_img)
    result = mxq_model.postprocess(output)

    assert result is not None
    assert result.task == "image_classification"
    assert result.acc is not None
    assert result.output is not None
