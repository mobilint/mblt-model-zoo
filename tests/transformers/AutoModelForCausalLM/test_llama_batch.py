from pprint import pprint

import pytest

from mblt_model_zoo.transformers import AutoTokenizer, pipeline
from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)


@pytest.fixture
def pipe():
    model_path = "mobilint/Llama-3.1-8B-Instruct-Batch16"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
    )
    yield pipe
    if isinstance(pipe.model, MobilintLlamaBatchForCausalLM):
        pipe.model.dispose()


def test_llama(pipe):
    pipe.generation_config.max_new_tokens = None

    messages = [
        [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
        ],
        [
            {"role": "system", "content": "You are Shakespeare."},
            {"role": "user", "content": "Write a short poem about coding."},
        ],
    ]

    outputs = pipe(
        messages,
        batch_size=2,
        max_new_tokens=30,
    )

    print("\n--- Batch Result 1 (Pirate) ---")
    print(outputs[0][0]["generated_text"][-1]["content"])

    print("\n--- Batch Result 2 (Shakespeare) ---")
    print(outputs[1][0]["generated_text"][-1]["content"])
