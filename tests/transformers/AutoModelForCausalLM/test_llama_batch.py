import pytest
from utils import BatchTextStreamer

from mblt_model_zoo.transformers import AutoTokenizer, pipeline
from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)

MODEL_PATHS = (
    "mobilint/Llama-3.2-3B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch32",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request):
    model_path = request.param

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
                "content": "You are a pirate chatbot with pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
        ],
        [
            {"role": "system", "content": "You are Shakespeare."},
            {"role": "user", "content": "Write a short poem about coding."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, my name is James."},
        ],
        [
            {"role": "system", "content": "You are James."},
            {"role": "user", "content": "Hi, my name is John."},
        ],
    ] * 4

    batch_size = len(messages)

    outputs = pipe(
        messages,
        batch_size=batch_size,
        chunk_size=16,
        max_new_tokens=512,
        streamer=BatchTextStreamer(
            tokenizer=pipe.tokenizer, batch_size=batch_size, skip_prompt=False
        ),
    )
