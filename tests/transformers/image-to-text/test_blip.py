import requests
from PIL import Image
from transformers import TextStreamer

MODEL_PATHS = ("mobilint/blip-image-captioning-large",)


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
