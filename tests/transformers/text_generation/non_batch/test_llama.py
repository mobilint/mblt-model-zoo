"""Non-batch tests for Llama instruct models."""

MODEL_PATHS = (
    "mobilint/Llama-3.2-1B-Instruct",
    "mobilint/Llama-3.2-3B-Instruct",
    "mobilint/Llama-3.1-8B-Instruct",
)


def test_llama(pipe, generation_token_limit: int) -> None:
    """Run a basic prompt against Llama instruct models."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    pipe(messages, max_new_tokens=generation_token_limit)
