"""Smoke tests for Mobilint Qwen2 EAGLE-3 chat generation."""

from __future__ import annotations

MODEL_PATHS = ("mobilint/EAGLE3-JPharmatron-7B",)


def test_qwen2_eagle3_multi_turn_chat(pipe, generation_token_limit: int) -> None:
    """Verify multi-turn chat runs without errors and returns structured outputs."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    first_turn_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Explain speculative decoding briefly.",
        },
    ]

    first_turn_outputs = pipe(first_turn_messages, max_new_tokens=generation_token_limit)

    assert isinstance(first_turn_outputs, list)
    assert len(first_turn_outputs) > 0
    assert isinstance(first_turn_outputs[0], dict)
    assert "generated_text" in first_turn_outputs[0]

    generated_chat = first_turn_outputs[0]["generated_text"]
    assert isinstance(generated_chat, list)
    assert len(generated_chat) >= len(first_turn_messages)

    second_turn_messages = generated_chat + [
        {
            "role": "user",
            "content": "Now compare it with greedy decoding in one sentence.",
        }
    ]

    second_turn_outputs = pipe(second_turn_messages, max_new_tokens=generation_token_limit)

    assert isinstance(second_turn_outputs, list)
    assert len(second_turn_outputs) > 0
    assert isinstance(second_turn_outputs[0], dict)
    assert "generated_text" in second_turn_outputs[0]
