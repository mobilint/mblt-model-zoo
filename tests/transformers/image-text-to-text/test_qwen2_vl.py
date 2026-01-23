import pytest
from transformers import TextStreamer, AutoProcessor, pipeline, AutoTokenizer


MODEL_PATHS = ("mobilint/Qwen2-VL-2B-Instruct",)


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


def test_qwen2_vl(pipe):
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
            "max_length": 512,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
