"""Non-batch tests for Cohere text-generation models."""

MODEL_PATHS = ("mobilint/c4ai-command-r7b-12-2024",)


def test_cohere2(pipe, generation_token_limit: int) -> None:
    """Run a basic prompt against Cohere Command R."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    pipe(messages, max_new_tokens=generation_token_limit)
