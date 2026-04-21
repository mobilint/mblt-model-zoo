"""Shared fixtures for batch text-generation tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
from transformers import AutoTokenizer, pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import BatchTextStreamer  # noqa: E402

from tests.npu_backend_options import BaseNpuParams


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared batch pipeline fixture from module-level model paths."""
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
    base_npu_params: BaseNpuParams,
):
    """Create a batch-capable text-generation pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = base_npu_params.base

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if model_kwargs:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
        )

    yield pipe
    del pipe


@pytest.fixture
def run_batch_generation():
    """Run generation for a list of batched chat messages."""

    def _run(pipe, messages: list[list[dict[str, str]]], max_new_tokens: int = 256) -> None:
        pipe.generation_config.max_new_tokens = None
        batch_size = len(messages)
        pipe(
            messages,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            streamer=BatchTextStreamer(
                tokenizer=pipe.tokenizer,
                batch_size=batch_size,
                skip_prompt=False,
            ),
        )

    return _run
