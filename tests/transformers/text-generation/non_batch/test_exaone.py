"""Non-batch tests for EXAONE instruct models."""

MODEL_PATHS = (
    "mobilint/EXAONE-3.5-2.4B-Instruct",
    "mobilint/EXAONE-3.5-7.8B-Instruct",
)


def test_exaone(pipe) -> None:
    """Run a basic prompt against EXAONE instruct models."""
    pipe.generation_config.max_new_tokens = None

    prompt = "너는 어떤 점에서 유용한지 짧게 설명해줘."

    messages = [
        {
            "role": "system",
            "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    pipe(messages, max_length=512)
