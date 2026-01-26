import pytest
from transformers import TextStreamer, AutoProcessor, pipeline


MODEL_PATHS = ("mobilint/aya-vision-8b",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path):
    model_path = request.param
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    
    if mxq_path:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
            model_kwargs={"mxq_path": mxq_path},
        )
    else:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
        )
    yield pipe
    del pipe


def test_aya(pipe):
    pipe.generation_config.max_new_tokens = None

    # Format message with the aya-vision chat template
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo=",
                },
                {"type": "text", "text": "Which one is shown in this picture?"},
            ],
        }
    ]

    pipe(
        text=messages,
        generate_kwargs={
            "max_length": 512,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
