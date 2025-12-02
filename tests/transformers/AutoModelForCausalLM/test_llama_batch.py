import pytest
from utils import BatchTextStreamer

from mblt_model_zoo.transformers import AutoTokenizer, pipeline
from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)
from mblt_model_zoo.transformers.utils.cache_utils import MobilintBatchCache


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
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Could you explain about LLM?"},
        ],
    ]

    batch_size = len(messages)
    past_key_values = MobilintBatchCache(
        mxq_model=pipe.model.get_cache_mxq_model(),
        batch_size=pipe.model.config.max_batch_size,
    )

    conversations = pipe(
        messages,
        batch_size=batch_size,
        max_new_tokens=512,
        past_key_values=past_key_values,
        streamer=BatchTextStreamer(
            tokenizer=pipe.tokenizer, batch_size=batch_size, skip_prompt=False
        ),
    )

    output = pipe(
        conversations,
        batch_size=batch_size,
        max_new_tokens=512,
        past_key_values=past_key_values,
        streamer=BatchTextStreamer(
            tokenizer=pipe.tokenizer, batch_size=batch_size, skip_prompt=False
        ),
    )
