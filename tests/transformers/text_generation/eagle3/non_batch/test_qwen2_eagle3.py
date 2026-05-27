"""Non-batch smoke test for Mobilint Qwen2 EAGLE-3 models."""

from __future__ import annotations

MODEL_PATHS = ("mobilint/EAGLE3-JPharmatron-7B",)


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
