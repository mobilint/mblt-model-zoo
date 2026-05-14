import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

MODEL_PATHS = (
    "mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "mobilint/EXAONE-Deep-2.4B",
    "mobilint/EXAONE-Deep-7.8B",
    "mobilint/EXAONE-3.5-2.4B-Instruct",
    "mobilint/EXAONE-3.5-7.8B-Instruct",
    "mobilint/EXAONE-4.0-1.2B",
    "mobilint/Llama-3.2-1B-Instruct",
    "mobilint/Llama-3.2-3B-Instruct",
    "mobilint/Llama-3.1-8B-Instruct",
    "mobilint/Qwen2.5-0.5B-Instruct",
    "mobilint/Qwen2.5-1.5B-Instruct",
    "mobilint/Qwen2.5-3B-Instruct",
    "mobilint/Qwen2.5-7B-Instruct",
    "mobilint/Qwen3-0.6B",
    "mobilint/Qwen3-1.7B",
    "mobilint/Qwen3-4B",
    "mobilint/Qwen3-8B",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def model_path(request):
    return request.param


@pytest.fixture(scope="module")
def model(model_path, revision, base_npu_params):
    model_kwargs = base_npu_params.base
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
        **model_kwargs,
    )
    yield model
    del model


@pytest.fixture(scope="module")
def tokenizer(model_path, revision):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )
    yield tokenizer


def test_cache(model, tokenizer, generation_token_limit: int):
    """Match fresh-prefill generation against a cache dump/load roundtrip."""
    model.generation_config.max_new_tokens = None
    model.generation_config.max_length = None

    prefix_messages = [
        {
            "role": "system",
            "content": "You are an AI assistant who is good at remembering people's name.",
        },
        {
            "role": "user",
            "content": (
                "My name is James. You should remember my name. "
                'If I ask "What is my name?", you should answer "Your name is James."'
            ),
        },
        {
            "role": "assistant",
            "content": "Your name is James.",
        },
    ]
    full_messages = prefix_messages + [{"role": "user", "content": "What is my name?"}]

    past_key_values = MobilintCache(model.get_cache_mxq_model())

    prefix_input_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=False,
    )
    full_input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=False,
    )
    prefix_length = prefix_input_ids.shape[1]
    assert torch.equal(full_input_ids[:, :prefix_length], prefix_input_ids)

    with torch.no_grad():
        model(
            prefix_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
        )

    past_key_values.dump_cache_memory()

    reference_cache = MobilintCache(model.get_cache_mxq_model())
    reference_output_ids = model.generate(
        full_input_ids,
        use_cache=True,
        past_key_values=reference_cache,
        do_sample=False,
        max_new_tokens=generation_token_limit,
    )

    model.dispose()
    model.launch()

    past_key_values.load_cache_memory()
    restored_output_ids = model.generate(
        full_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        do_sample=False,
        max_new_tokens=generation_token_limit,
    )

    reference_generated_ids = reference_output_ids[:, full_input_ids.shape[-1] :]
    restored_generated_ids = restored_output_ids[:, full_input_ids.shape[-1] :]

    assert torch.equal(restored_generated_ids, reference_generated_ids)
