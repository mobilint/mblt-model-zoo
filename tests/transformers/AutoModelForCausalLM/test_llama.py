import pytest
from transformers import TextStreamer

from mblt_model_zoo.hf_transformers import AutoTokenizer, pipeline

MODEL_PATHS = (
    "mobilint/Llama-3.2-1B-Instruct",
    "mobilint/Llama-3.2-3B-Instruct",
    "mobilint/Llama-3.1-8B-Instruct",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path):
    model_path = request.param

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if mxq_path:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            model_kwargs={"mxq_path": mxq_path},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
        )
    yield pipe
    pipe.model.dispose()


def test_llama(pipe):
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
