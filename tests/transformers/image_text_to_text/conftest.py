"""Shared fixtures for image-text-to-text tests."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoProcessor, pipeline

from tests.npu_backend_options import VisionTextNpuParams


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pytest.fixture(scope="module")
def pipe(
    request: pytest.FixtureRequest,
    revision: Optional[str],
    vision_text_npu_params: VisionTextNpuParams,
):
    """Create an image-text-to-text pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = {**vision_text_npu_params.vision, **vision_text_npu_params.text}
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
    )

    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        processor=processor,
        trust_remote_code=True,
        revision=revision,
        model_kwargs=model_kwargs or None,
    )

    yield pipe
    del pipe
