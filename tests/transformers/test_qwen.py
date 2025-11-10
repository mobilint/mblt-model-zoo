import pytest
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


@pytest.fixture
def pipe():
    model_path = "mobilint/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    )
    yield pipe
    pipe.model.dispose()


def test_llama(pipe):
    pipe.generation_config.max_new_tokens = None

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    outputs = pipe(
        messages,
        max_length=512,
    )
