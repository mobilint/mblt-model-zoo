"""Batch tests for Llama instruct models."""

MODEL_PATHS = (
    "mobilint/Llama-3.2-1B-Instruct-Batch16",
    "mobilint/Llama-3.2-1B-Instruct-Batch32",
    "mobilint/Llama-3.2-3B-Instruct-Batch16",
    "mobilint/Llama-3.2-3B-Instruct-Batch32",
    "mobilint/Llama-3.1-8B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch32",
)


def test_llama_batch(pipe, run_batch_generation) -> None:
    """Run short prompts against Llama batch models."""
    messages = [
        [
            {
                "role": "system",
                "content": "You are a pirate chatbot with pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
        ],
        [
            {"role": "system", "content": "You are Shakespeare."},
            {"role": "user", "content": "Write a short poem about coding."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, my name is James."},
        ],
        [
            {"role": "system", "content": "You are James."},
            {"role": "user", "content": "Hi, my name is John."},
        ],
        [
            {"role": "system", "content": "You are a travel planner."},
            {"role": "user", "content": "Plan a one-day walkable tour of Rome."},
        ],
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain the Pythagorean theorem simply."},
        ],
        [
            {"role": "system", "content": "You are a nutritionist."},
            {"role": "user", "content": "Suggest quick high-protein breakfast ideas."},
        ],
        [
            {"role": "system", "content": "You are a coding mentor."},
            {"role": "user", "content": "Show how to reverse a string in Python."},
        ],
        [
            {"role": "system", "content": "You are a science writer."},
            {"role": "user", "content": "Describe why the sky appears blue to kids."},
        ],
        [
            {"role": "system", "content": "You are a fitness coach."},
            {"role": "user", "content": "Design a 20 minute bodyweight workout."},
        ],
        [
            {"role": "system", "content": "You are a product manager."},
            {
                "role": "user",
                "content": "Draft bullet points for release notes on a new dark mode.",
            },
        ],
        [
            {"role": "system", "content": "You are a translator."},
            {
                "role": "user",
                "content": "Translate 'Bonjour, je m'appelle Lina' to English.",
            },
        ],
        [
            {"role": "system", "content": "You are a storyteller."},
            {
                "role": "user",
                "content": "Tell a short bedtime story about a brave cat.",
            },
        ],
        [
            {"role": "system", "content": "You are a data analyst."},
            {
                "role": "user",
                "content": "List a few KPIs for an online bookstore.",
            },
        ],
        [
            {"role": "system", "content": "You are a security expert."},
            {"role": "user", "content": "Give tips for creating strong passwords."},
        ],
        [
            {"role": "system", "content": "You are a debate moderator."},
            {
                "role": "user",
                "content": "Pose a neutral question about renewable energy policy.",
            },
        ],
    ]
    run_batch_generation(pipe, messages)
