"""Non-batch tests for EXAONE 4 models."""

import random

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATHS = ("mobilint/EXAONE-4.0-1.2B",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def model(request, revision, base_npu_params):
    model_path = request.param
    model_kwargs = base_npu_params.base
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
        **model_kwargs,
    )
    yield model
    del model


@pytest.fixture(params=MODEL_PATHS, scope="module")
def tokenizer(request, revision):
    model_path = request.param
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )
    yield tokenizer


def test_exaone4(model, tokenizer, generation_token_limit: int) -> None:
    """Run representative EXAONE 4 generation modes."""
    model.generation_config.max_new_tokens = None
    model.generation_config.max_length = None
    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=False)

    print("\n - Non-reasoning mode\n")

    prompt = "Explain how wonderful you are"

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )

    model.generate(
        input_ids.to(model.device),
        max_new_tokens=generation_token_limit,
        do_sample=False,
        streamer=streamer,
    )

    print("\n - Reasoning mode\n")

    messages = [{"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        enable_thinking=True,
    )

    model.generate(
        input_ids.to(model.device),
        max_new_tokens=generation_token_limit,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=streamer,
    )

    print("\n - Agentic tool use\n")

    def roll_dice(max_num: int):
        return random.randint(1, max_num)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "roll_dice",
                "description": "Roll a dice with the number 1 to N. User can select the number N.",
                "parameters": {
                    "type": "object",
                    "required": ["max_num"],
                    "properties": {
                        "max_num": {
                            "type": "int",
                            "description": "Max number of the dice",
                        }
                    },
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "Roll D6 dice twice!"}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        tools=tools,
    )

    model.generate(
        input_ids.to(model.device),
        max_new_tokens=generation_token_limit,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=streamer,
    )
