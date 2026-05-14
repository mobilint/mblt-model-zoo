"""Batch tests for Qwen3 models."""

MODEL_PATHS = (
    "mobilint/Qwen3-0.6B-Batch16",
    "mobilint/Qwen3-1.7B-Batch16",
    "mobilint/Qwen3-4B-Batch16",
    "mobilint/Qwen3-8B-Batch16",
)


def test_qwen3_batch(pipe, run_batch_generation) -> None:
    """Run short English prompts against Qwen3 batch models."""
    messages = [
        [{"role": "user", "content": "Explain machine learning in one sentence."}],
        [{"role": "user", "content": "What is a function in Python?"}],
        [{"role": "user", "content": "Give one idea for a team building activity."}],
        [{"role": "user", "content": "What is the purpose of unit tests?"}],
        [{"role": "user", "content": "Write a short status update."}],
        [{"role": "user", "content": "What is an operating system?"}],
        [{"role": "user", "content": "Suggest one way to reduce meeting time."}],
        [{"role": "user", "content": "What is data preprocessing?"}],
        [{"role": "user", "content": "Name one good habit for remote work."}],
        [{"role": "user", "content": "What is recursion?"}],
        [{"role": "user", "content": "Give one tip for writing clearer prompts."}],
        [{"role": "user", "content": "What is a GPU used for?"}],
        [{"role": "user", "content": "Write a short product description."}],
        [{"role": "user", "content": "What is encryption?"}],
        [{"role": "user", "content": "Suggest one beginner coding project."}],
        [{"role": "user", "content": "What is a tokenizer?"}],
    ]
    run_batch_generation(pipe, messages)
