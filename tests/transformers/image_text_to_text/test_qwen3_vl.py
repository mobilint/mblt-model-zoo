from transformers import TextStreamer

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = (
    "mobilint/Qwen3-VL-2B-Instruct",
    "mobilint/Qwen3-VL-4B-Instruct",
)


def test_qwen3_vl(pipe, generation_token_limit: int):
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
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
