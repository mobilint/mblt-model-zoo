"""Smoke tests for Mobilint Qwen2 EAGLE-3 chat generation."""

from __future__ import annotations

import torch
from transformers import TextStreamer

MODEL_PATHS = ("mobilint/EAGLE3-JPharmatron-7B",)


def test_qwen2_eagle3_multi_turn_chat(pipe, generation_token_limit: int) -> None:
    """Verify multi-turn chat reuses KV cache through direct `generate` calls."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    streamer = TextStreamer(tokenizer, skip_prompt=False)

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
    first_turn_prompt = tokenizer.apply_chat_template(
        first_turn_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    first_turn_inputs = tokenizer(first_turn_prompt, return_tensors="pt")
    input_ids = first_turn_inputs["input_ids"]

    first_turn_outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=generation_token_limit,
        streamer=streamer,
        return_dict_in_generate=True,
    )

    assert hasattr(first_turn_outputs, "sequences")
    assert hasattr(first_turn_outputs, "past_key_values")
    first_turn_sequences = first_turn_outputs.sequences
    first_turn_cache = first_turn_outputs.past_key_values
    assert isinstance(first_turn_sequences, torch.Tensor)
    assert first_turn_sequences.shape[1] > input_ids.shape[1]
    assert first_turn_cache is not None

    first_turn_generated_ids = first_turn_sequences[:, input_ids.shape[1] :]
    first_turn_generated_text = tokenizer.decode(first_turn_generated_ids[0], skip_special_tokens=True).strip()
    assert first_turn_generated_text != ""

    second_turn_messages = first_turn_messages + [
        {
            "role": "assistant",
            "content": first_turn_generated_text,
        },
        {
            "role": "user",
            "content": "Now compare it with greedy decoding in one sentence.",
        },
    ]
    second_turn_prompt = tokenizer.apply_chat_template(
        second_turn_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    second_turn_inputs = tokenizer(second_turn_prompt, return_tensors="pt")
    second_turn_input_ids = second_turn_inputs["input_ids"]
    second_turn_delta_input_ids = second_turn_input_ids[:, input_ids.shape[1] :]

    second_turn_outputs = model.generate(
        input_ids=second_turn_delta_input_ids,
        past_key_values=first_turn_cache,
        max_new_tokens=generation_token_limit,
        streamer=streamer,
        return_dict_in_generate=True,
    )

    assert hasattr(second_turn_outputs, "sequences")
    assert hasattr(second_turn_outputs, "past_key_values")
    second_turn_sequences = second_turn_outputs.sequences
    assert isinstance(second_turn_sequences, torch.Tensor)
    assert second_turn_sequences.shape[1] > second_turn_delta_input_ids.shape[1]
    assert second_turn_outputs.past_key_values is first_turn_cache
