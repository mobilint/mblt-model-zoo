import pytest
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


MODEL_PATHS = (
    "mobilint/EXAONE-3.5-2.4B-Instruct",
    "mobilint/EXAONE-3.5-7.8B-Instruct",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request):
    model_path = request.param

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    )
    yield pipe
    pipe.model.dispose()


def test_exaone(pipe):
    pipe.generation_config.max_new_tokens = None

    # Choose your prompt
    prompt = "Explain how wonderful you are"  # English example
    prompt = "스스로를 자랑해 봐"  # Korean example

    messages = [
        {
            "role": "system",
            "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(
        messages,
        max_length=512,
    )
