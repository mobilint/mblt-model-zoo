"""Batch NPU end-to-end video tests for Qwen3-VL."""

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

MODEL_PATHS = ("./Qwen3-VL-8B-Instruct-Batch16",)

VIDEO_URL = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_1MB.mp4"

USER_QUESTIONS = [
    "Describe this video.",
    "What is the main subject of the video?",
    "How would you characterize the motion in this video?",
    "What colors dominate the video?",
    "Is this taken indoors or outdoors?",
    "What mood does the video convey?",
    "How many distinct subjects appear?",
    "Describe the lighting throughout the video.",
    "What is happening at the beginning of the video?",
    "What is happening at the end of the video?",
    "Is there any text or signage visible?",
    "What environment or setting is shown?",
    "Summarize the video in five words.",
    "What emotion does this video evoke?",
    "Suggest a short caption for this video.",
    "What kind of camera work is used?",
]


def _conversation(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_URL},
                {"type": "text", "text": question},
            ],
        }
    ]


def test_qwen3_vl_video_batch(pipe, run_batch_generation) -> None:
    """Run a batched video conversation set through the Qwen3-VL pipeline."""
    conversations = [_conversation(q) for q in USER_QUESTIONS]
    run_batch_generation(pipe, conversations)
