"""Batch NPU end-to-end tests for Qwen3-VL."""

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = ("mobilint/Qwen3-VL-8B-Instruct-Batch16",)


def test_qwen3_vl_batch(pipe, run_batch_generation) -> None:
    """Run a batched image-and-text conversation set through the Qwen3-VL pipeline."""
    conversations = [
        [
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
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "What is the main subject?"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "List two colors you see."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Is this taken indoors or outdoors?"},
                ],
            }
        ],
    ]
    run_batch_generation(pipe, conversations)
