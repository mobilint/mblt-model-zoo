"""Non-batch tests for Qwen3 models."""

MODEL_PATHS = (
    "mobilint/Qwen3-0.6B",
    "mobilint/Qwen3-1.7B",
    "mobilint/Qwen3-4B",
    "mobilint/Qwen3-8B",
)


def test_qwen2(pipe) -> None:
    """Run a basic prompt against Qwen3."""
    pipe.generation_config.max_new_tokens = None

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]

    pipe(messages, max_length=512)
