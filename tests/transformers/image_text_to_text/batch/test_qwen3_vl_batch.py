"""Batch NPU end-to-end tests for Qwen3-VL."""

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = ("mobilint/Qwen3-VL-8B-Instruct-Batch16",)

IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

USER_QUESTIONS = [
    "Describe this image.",
    "What is the main subject?",
    "List two colors you see.",
    "Is this taken indoors or outdoors?",
    "What is the mood of this scene?",
    "How many people are in the image?",
    "What time of day does it appear to be?",
    "Are there any animals visible?",
    "What is in the background?",
    "Describe the lighting in one sentence.",
    "Is there any text or signage present?",
    "What kind of weather is depicted?",
    "Summarize the image in five words.",
    "What emotion does this image evoke?",
    "Is this a photograph or an illustration?",
    "Suggest a short caption for this image.",
]


def _conversation(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_URL},
                {"type": "text", "text": question},
            ],
        }
    ]


def test_qwen3_vl_batch(pipe, run_batch_generation) -> None:
    """Run a batched image-and-text conversation set through the Qwen3-VL pipeline."""
    conversations = [_conversation(q) for q in USER_QUESTIONS]
    run_batch_generation(pipe, conversations)
