import pytest
from transformers import TextStreamer

from tests.npu_backend_options import VisionTextNpuParams, option_value_was_provided

MODEL_PATHS = ("mobilint/Qwen2-VL-2B-Instruct",)


def _has_explicit_npu_overrides(config: pytest.Config) -> bool:
    """Return whether the current invocation explicitly requested Qwen2-VL NPU overrides."""
    if option_value_was_provided(config, "", "core_mode"):
        return True

    option_names = ("mxq_path", "dev_no", "core_mode", "target_cores", "target_clusters")
    return any(
        option_value_was_provided(config, prefix, option_name)
        for prefix in ("vision", "text")
        for option_name in option_names
    )

def build_model_kwargs(
    request: pytest.FixtureRequest,
    vision_text_npu_params: VisionTextNpuParams,
) -> dict[str, object]:
    """Return backend kwargs for the shared Qwen2-VL pipeline fixture."""
    return (
        {**vision_text_npu_params.vision, **vision_text_npu_params.text}
        if _has_explicit_npu_overrides(request.config)
        else {}
    )


def test_qwen2_vl(pipe, generation_token_limit: int):
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
