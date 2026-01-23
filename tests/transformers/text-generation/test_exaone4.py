import random

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin

MODEL_PATHS = ("mobilint/EXAONE-4.0-1.2B",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def model(request, mxq_path):
    model_path = request.param
    if mxq_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            mxq_path=mxq_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    yield model
    del model


@pytest.fixture(params=MODEL_PATHS, scope="module")
def tokenizer(request):
    model_path = request.param
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    yield tokenizer


def test_exaone4(model, tokenizer):
    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=False)

    print("\n - Non-reasoning mode\n")

    # Choose your prompt
    prompt = "Explain how wonderful you are"
    # prompt = "Explica lo increíble que eres"
    # prompt = "너가 얼마나 대단한지 설명해 봐"

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    model.generate(
        input_ids.to(model.device),
        max_length=512,
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
        enable_thinking=True,
    )

    model.generate(
        input_ids.to(model.device),
        max_length=512,
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
        tools=tools,
    )

    model.generate(
        input_ids.to(model.device),
        max_length=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=streamer,
    )
