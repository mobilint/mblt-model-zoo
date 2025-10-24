import pytest
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


@pytest.fixture
def pipe():
    model_path = "mobilint/c4ai-command-r7b-12-2024"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "text-generation",
        model=model_path,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    )
    yield pipe
    pipe.model.dispose()


def test_cohere2(pipe):
    pipe.generation_config.max_new_tokens = None

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipe(
        messages,
        max_length=512,
    )
