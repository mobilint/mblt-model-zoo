import pytest
import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer, pipeline

MODEL_PATHS = ("mobilint/blip-image-captioning-large",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, vision_text_npu_params):
    model_path = request.param
    model_kwargs = {**vision_text_npu_params.vision, **vision_text_npu_params.text}

    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=revision,
    )
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        processor=processor,
        trust_remote_code=True,
        revision=revision,
        model_kwargs=model_kwargs or None,
    )
    yield pipe
    del pipe


def test_blip(pipe, generation_token_limit: int):
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # conditional image captioning
    text = "a photography of"
    pipe(
        raw_image,
        text,
        generate_kwargs={
            "max_new_tokens": generation_token_limit,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )

    # unconditional image captioning
    pipe(
        raw_image,
        "",
        generate_kwargs={
            "max_new_tokens": generation_token_limit,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
