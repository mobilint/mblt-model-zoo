import pytest
from utils import BatchTextStreamer

from mblt_model_zoo.transformers import AutoTokenizer, pipeline
from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)

MODEL_PATHS = (
    "mobilint/Llama-3.2-3B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch32",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request):
    model_path = request.param

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
    )
    yield pipe
    if isinstance(pipe.model, MobilintLlamaBatchForCausalLM):
        pipe.model.dispose()


def test_llama(pipe):
    pipe.generation_config.max_new_tokens = None

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

    batch_size = len(messages)

    outputs = pipe(
        messages,
        batch_size=batch_size,
        chunk_size=16,
        max_new_tokens=512,
        streamer=BatchTextStreamer(
            tokenizer=pipe.tokenizer,
            batch_size=batch_size,
            skip_prompt=False,
            alert_token_length=[256, 260, 270, 280],
        ),
    )
