import pytest
from transformers import AutoProcessor, TextStreamer, pipeline

MODEL_PATHS = (
    "mobilint/Qwen3-VL-2B-Instruct",
    "mobilint/Qwen3-VL-4B-Instruct",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, vision_text_npu_params):
    model_path = request.param
    model_kwargs = {**vision_text_npu_params.vision, **vision_text_npu_params.text}
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
    )

    if model_kwargs:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            processor=processor,
            trust_remote_code=True,
            revision=revision,
        )
    yield pipe
    del pipe


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
