from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoProcessor


def test_qwen2_vl():
    model_name = "mobilint/Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        processor=processor,
    )
    pipe.generation_config.max_new_tokens = None

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
            "max_length": 4096,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
