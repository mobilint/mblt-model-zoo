"""Batch tests for Qwen2.5 models."""

MODEL_PATHS = (
    "mobilint/Qwen2.5-0.5B-Instruct-Batch16",
    "mobilint/Qwen2.5-1.5B-Instruct-Batch16",
    "mobilint/Qwen2.5-3B-Instruct-Batch16",
    "mobilint/Qwen2.5-7B-Instruct-Batch16",
)


def test_qwen_batch(pipe, run_batch_generation) -> None:
    """Run short English prompts against Qwen2.5 batch models."""
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain large language models in one sentence."},
        ],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Name one practical use of AI in education."},
        ],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "What is a database?"}],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Write a short welcome message."}],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Give one tip for learning Python."},
        ],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "What does CPU stand for?"}],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Suggest a healthy snack."}],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "What is version control?"}],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Name one benefit of regular exercise."},
        ],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Explain debugging briefly."}],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "What is an API?"}],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Give one time management tip."}],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is cloud storage?"},
        ],
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Write a short thank-you note."}],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is a neural network?"},
        ],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Suggest one idea for a weekend hobby."},
        ],
    ]
    run_batch_generation(pipe, messages)
