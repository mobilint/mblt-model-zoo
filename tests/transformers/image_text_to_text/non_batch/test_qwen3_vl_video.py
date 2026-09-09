from transformers import TextStreamer

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_static_vision,
    skip_if_transformers_lacks_qwen3_vl_support,
    skip_qwen3_vl_pipeline_if_tf_5_4,
)

skip_if_transformers_lacks_qwen3_vl_support()
skip_qwen3_vl_pipeline_if_tf_5_4()

# 8B dropped: see QWEN3_VL_8B_MXQ_INCOMPATIBLE_REASON in qwen3_vl_compat.
MODEL_PATHS = (
    "mobilint/Qwen3-VL-2B-Instruct",
    "mobilint/Qwen3-VL-4B-Instruct",
)

VIDEO_URL = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4"


def test_qwen3_vl_video(pipe, generation_token_limit: int):
    skip_if_static_vision(pipe, "Video input")

    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_URL},
                {"type": "text", "text": "Describe this video."},
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
