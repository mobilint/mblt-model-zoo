import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

MODEL_PATHS = ("mobilint/c4ai-command-r7b-12-2024",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path):
    model_path = request.param
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if mxq_path:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            model_kwargs={"mxq_path": mxq_path},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
        )
    yield pipe
    del pipe


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
