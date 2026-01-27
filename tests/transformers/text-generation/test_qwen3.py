import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

MODEL_PATHS = (
    "mobilint/Qwen3-0.6B",
    "mobilint/Qwen3-1.7B",
    "mobilint/Qwen3-4B",
    "mobilint/Qwen3-8B",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path, revision, embedding_weight):
    model_path = request.param
    model_kwargs = {}
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
    if embedding_weight:
        model_kwargs["embedding_weight"] = embedding_weight

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if model_kwargs:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
        )
    yield pipe
    del pipe


def test_qwen2(pipe):
    pipe.generation_config.max_new_tokens = None

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(
        messages,
        max_length=512,
    )
