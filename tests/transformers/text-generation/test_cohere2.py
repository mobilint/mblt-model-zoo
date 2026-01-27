import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

MODEL_PATHS = ("mobilint/c4ai-command-r7b-12-2024",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, npu_params):
    model_path = request.param
    npu_params.warn_unused({"base"})
    model_kwargs = npu_params.base
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


def test_cohere2(pipe):
    pipe.generation_config.max_new_tokens = None

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    outputs = pipe(
        messages,
        max_length=512,
    )
