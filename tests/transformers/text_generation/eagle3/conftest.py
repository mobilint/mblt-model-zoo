"""Shared fixtures for EAGLE-3 text-generation tests."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pytest.fixture(scope="module")
def pipe(request: pytest.FixtureRequest, revision: Optional[str], eagle3_npu_params):
    """Create an EAGLE-3 text-generation pipeline for the parametrized model."""
    model_path = request.param
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )
    model_kwargs = eagle3_npu_params.model
    if model_kwargs:
        created_pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        created_pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
        )
    yield created_pipe
    del created_pipe
