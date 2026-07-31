"""Batch NPU end-to-end multi-image tests for Qwen3-VL."""

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = ("mobilint/Qwen3-VL-8B-Instruct-Batch16",)

IMAGE_URLS = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640",
)

USER_QUESTIONS = [
    "Compare these images. What are the differences?",
    "Which image looks more colorful?",
    "What subjects appear in each image?",
    "Which image is taken indoors vs outdoors?",
    "Describe the lighting in each image.",
    "How many people or animals appear in each?",
    "What emotions do the two images evoke?",
    "Summarize each image in one sentence.",
    "Which image is more visually complex?",
    "What time of day does each image depict?",
    "Are there any common elements between the two?",
    "Which image would you prefer as a wallpaper?",
    "Describe the color palette of each image.",
    "Which image has a more dynamic composition?",
    "Suggest a caption for each image.",
    "What story could connect these two images?",
]


def _conversation(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": url} for url in IMAGE_URLS),
                {"type": "text", "text": question},
            ],
        }
    ]


def test_qwen3_vl_multi_image_batch(pipe, run_batch_generation) -> None:
    """Run a batched multi-image conversation set through the Qwen3-VL pipeline."""
    conversations = [_conversation(q) for q in USER_QUESTIONS]
    run_batch_generation(pipe, conversations)
