from typing import Any, Optional

import pytest
from transformers import AutoTokenizer, pipeline
from utils import BatchTextStreamer

MODEL_PATHS = (
    "mobilint/Llama-3.2-1B-Instruct-Batch16",
    "mobilint/Llama-3.2-1B-Instruct-Batch32",
    "mobilint/Llama-3.2-3B-Instruct-Batch16",
    "mobilint/Llama-3.2-3B-Instruct-Batch32",
    "mobilint/Llama-3.1-8B-Instruct-Batch16",
    "mobilint/Llama-3.1-8B-Instruct-Batch32",
)


def _parse_target_cores(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [int(item.strip()) for item in text.split(";") if item.strip()]


def _build_model_kwargs(request: pytest.FixtureRequest, embedding_weight: Optional[str]) -> dict[str, Any]:
    config = request.config
    model_kwargs: dict[str, Any] = {}

    mxq_path = config.getoption("--mxq-path")
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path

    dev_no = config.getoption("--dev-no")
    if dev_no is not None:
        model_kwargs["dev_no"] = dev_no

    raw_core_mode = config.getoption("--core-mode")
    core_mode = None if raw_core_mode in {None, "", "all"} else raw_core_mode
    if core_mode:
        model_kwargs["core_mode"] = core_mode

    target_cores = _parse_target_cores(config.getoption("--target-cores"))
    if target_cores is not None:
        model_kwargs["target_cores"] = target_cores

    target_clusters = _parse_target_clusters(config.getoption("--target-clusters"))
    if target_clusters is not None:
        model_kwargs["target_clusters"] = target_clusters

    if core_mode == "single" and target_cores is None:
        model_kwargs["target_cores"] = ["0:0"]
    elif core_mode == "global4" and target_clusters is None:
        model_kwargs["target_clusters"] = [0]
    elif core_mode == "global8" and target_clusters is None:
        model_kwargs["target_clusters"] = [0, 1]

    if embedding_weight:
        model_kwargs["embedding_weight"] = embedding_weight

    return model_kwargs


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, embedding_weight):
    model_path = request.param
    model_kwargs = _build_model_kwargs(request, embedding_weight)

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    if model_kwargs:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
    yield pipe
    del pipe


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

    pipe(
        messages,
        batch_size=batch_size,
        max_new_tokens=256,
        streamer=BatchTextStreamer(
            tokenizer=pipe.tokenizer,
            batch_size=batch_size,
            skip_prompt=False,
        ),
    )
