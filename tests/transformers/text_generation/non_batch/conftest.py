"""Shared fixtures for non-batch text-generation tests."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

from tests.pipe_teardown import pipe_fixture


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pipe_fixture()
def pipe(request: pytest.FixtureRequest, revision: Optional[str], base_npu_params):
    """Create a text-generation pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = base_npu_params.base

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )

    if model_kwargs:
        return pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    return pipeline(
        "text-generation",
        model=model_path,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
        trust_remote_code=True,
        revision=revision,
    )
