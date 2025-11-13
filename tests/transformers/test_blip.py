import pytest
import requests
from PIL import Image
from transformers import TextStreamer

from mblt_model_zoo.transformers import AutoProcessor, pipeline


@pytest.fixture
def pipe():
    model_name = "mobilint/blip-image-captioning-large"

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        processor=processor,
    )
    yield pipe
    pipe.model.dispose()


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
