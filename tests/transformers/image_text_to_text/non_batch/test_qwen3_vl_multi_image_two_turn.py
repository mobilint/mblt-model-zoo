"""Non-batch two-turn multi-image test for the dynamic Qwen3-VL 8B release."""

import pytest
from transformers import TextStreamer

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from transformers.image_utils import load_image  # noqa: E402

QWEN3_VL_8B_MODEL = "mobilint/Qwen3-VL-8B-Instruct"
IMAGE_URLS = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640",
)


@pytest.fixture(scope="module")
def qwen3_vl_8b_model_and_processor(vision_text_npu_params, revision):
    """Load the 8B model and synchronize dynamic vision for MXQ overrides."""
    model_kwargs = {**vision_text_npu_params.vision, **vision_text_npu_params.text}
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN3_VL_8B_MODEL,
        trust_remote_code=True,
        revision=revision,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        QWEN3_VL_8B_MODEL,
        trust_remote_code=True,
        revision=revision,
    )

    if "vision_mxq_path" in model_kwargs:
        processor.sync_dynamic_vision_from_model(model)

    return model, processor


def test_qwen3_vl_8b_multi_image_two_turns(qwen3_vl_8b_model_and_processor, generation_token_limit: int):
    """Run the two-turn dynamic-vision flow used by the development tester."""
    model, processor = qwen3_vl_8b_model_and_processor
    if not processor.dynamic_vision:
        pytest.skip("The 8B model is static-vision; this test requires a dynamic-vision release.")

    images = [load_image(url) for url in IMAGE_URLS]

    def generate_response(messages):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_token_limit,
            streamer=TextStreamer(tokenizer=processor.tokenizer, skip_prompt=True),
        )
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        assert response
        return response

    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": url} for url in IMAGE_URLS),
                {"type": "text", "text": "Compare these images. What are the differences?"},
            ],
        }
    ]
    first_response = generate_response(messages)

    messages.extend(
        [
            {"role": "assistant", "content": first_response},
            {"role": "user", "content": "What else can you tell me about it?"},
        ]
    )
    generate_response(messages)
