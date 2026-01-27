import pytest
import requests
from PIL import Image
from transformers import TextStreamer, AutoProcessor, pipeline


MODEL_PATHS = ("mobilint/blip-image-captioning-large",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path, revision):
    model_path = request.param

    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        revision=revision,
    )
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        processor=processor,
        trust_remote_code=True,
        revision=revision,
    )
    yield pipe
    del pipe


def test_blip(pipe):
    pipe.generation_config.max_new_tokens = None

    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # conditional image captioning
    text = "a photography of"
    pipe(
        raw_image,
        text,
        generate_kwargs={
            "max_length": 512,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )

    # unconditional image captioning
    pipe(
        raw_image,
        "",
        generate_kwargs={
            "max_length": 512,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
