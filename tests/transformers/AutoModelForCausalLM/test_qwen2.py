import pytest
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


MODEL_PATHS = (
    "mobilint/Qwen2.5-0.5B-Instruct",
    "mobilint/Qwen2.5-1.5B-Instruct",
    "mobilint/Qwen2.5-3B-Instruct",
    "mobilint/Qwen2.5-7B-Instruct",
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


def test_qwen2(pipe):
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
