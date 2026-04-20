"""Non-batch tests for Cohere text-generation models."""

MODEL_PATHS = ("mobilint/c4ai-command-r7b-12-2024",)


def test_cohere2(pipe) -> None:
    """Run a basic prompt against Cohere Command R."""
    pipe.generation_config.max_new_tokens = None

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    pipe(messages, max_length=512)
