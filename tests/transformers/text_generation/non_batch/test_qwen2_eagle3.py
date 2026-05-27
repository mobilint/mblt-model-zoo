"""Non-batch smoke test for Mobilint Qwen2 EAGLE-3 models."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

MODEL_PATHS = ("mobilint/EAGLE3-JPharmatron-7B",)


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


def test_qwen2_eagle3(pipe, generation_token_limit: int) -> None:
    """Run a basic prompt against the EAGLE-3 Qwen2 model."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Explain speculative decoding briefly.",
        },
    ]

    pipe(messages, max_new_tokens=generation_token_limit)
