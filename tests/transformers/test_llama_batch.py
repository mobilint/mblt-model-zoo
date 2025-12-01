from pprint import pprint

import pytest

from mblt_model_zoo.transformers import AutoTokenizer, pipeline
from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)


@pytest.fixture
def pipe():
    model_path = "mobilint/Llama-3.2-3B-Instruct-Batch"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device_map="auto",
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
        max_length=512,
    )

    print("\n--- Batch Result 1 (Pirate) ---")
    pprint(outputs[0])

    print("\n--- Batch Result 2 (Shakespeare) ---")
    pprint(outputs[1])
