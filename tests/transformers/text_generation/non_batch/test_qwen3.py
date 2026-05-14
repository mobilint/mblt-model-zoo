"""Non-batch tests for Qwen3 models."""

MODEL_PATHS = (
    "mobilint/Qwen3-0.6B",
    "mobilint/Qwen3-1.7B",
    "mobilint/Qwen3-4B",
    "mobilint/Qwen3-8B",
)


def test_qwen2(pipe, generation_token_limit: int) -> None:
    """Run a basic prompt against Qwen3."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]

    pipe(messages, max_new_tokens=generation_token_limit)
