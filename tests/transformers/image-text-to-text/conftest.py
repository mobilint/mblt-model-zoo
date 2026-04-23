"""Shared fixtures for image-text-to-text tests."""

from __future__ import annotations

from typing import Any, Optional

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


def _default_model_kwargs(vision_text_npu_params: VisionTextNpuParams) -> dict[str, Any]:
    """Return the default merged vision and text backend kwargs."""
    return {**vision_text_npu_params.vision, **vision_text_npu_params.text}


def _resolve_model_kwargs(
    request: pytest.FixtureRequest,
    vision_text_npu_params: VisionTextNpuParams,
) -> dict[str, Any]:
    """Return module-specific backend kwargs for the shared pipeline fixture."""
    builder = getattr(request.module, "build_model_kwargs", None)
    if callable(builder):
        return builder(request, vision_text_npu_params)
    return _default_model_kwargs(vision_text_npu_params)


@pytest.fixture(scope="module")
def pipe(
    request: pytest.FixtureRequest,
    revision: Optional[str],
    vision_text_npu_params: VisionTextNpuParams,
):
    """Create an image-text-to-text pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = _resolve_model_kwargs(request, vision_text_npu_params)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
    )

    if model_kwargs:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
            revision=revision,
        )

    yield pipe
    del pipe
