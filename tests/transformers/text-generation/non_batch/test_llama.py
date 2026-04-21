"""Non-batch tests for Llama instruct models."""

MODEL_PATHS = (
    "mobilint/Llama-3.2-1B-Instruct",
    "mobilint/Llama-3.2-3B-Instruct",
    "mobilint/Llama-3.1-8B-Instruct",
)


def test_llama(pipe) -> None:
    """Run a basic prompt against Llama instruct models."""
    pipe.generation_config.max_new_tokens = None

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    pipe(messages, max_length=512)
