from transformers import TextStreamer

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_static_vision,
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = (
    "mobilint/Qwen3-VL-2B-Instruct",
    "mobilint/Qwen3-VL-4B-Instruct",
    "mobilint/Qwen3-VL-8B-Instruct",
)

IMAGE_URLS = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640",
)

def test_qwen3_vl_multi_image(pipe, generation_token_limit: int):
    skip_if_static_vision(pipe, "Multi-image input")

    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": url} for url in IMAGE_URLS),
                {"type": "text", "text": "Compare these images. What are the differences?"},
            ],
        }
    ]

    pipe(
        text=messages,
        generate_kwargs={
            "max_new_tokens": generation_token_limit,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
