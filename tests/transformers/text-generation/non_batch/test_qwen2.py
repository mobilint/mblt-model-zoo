"""Non-batch tests for Qwen2.5 models."""

MODEL_PATHS = (
    "mobilint/Qwen2.5-0.5B-Instruct",
    "mobilint/Qwen2.5-1.5B-Instruct",
    "mobilint/Qwen2.5-3B-Instruct",
    "mobilint/Qwen2.5-7B-Instruct",
)


def test_qwen2(pipe) -> None:
    """Run a basic prompt against Qwen2.5."""
    pipe.generation_config.max_new_tokens = None

    prompt = "Give me a short introduction to large language model."
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    pipe(messages, max_length=512)
