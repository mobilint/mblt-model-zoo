import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

MODEL_PATHS = (
    "mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "mobilint/EXAONE-Deep-2.4B",
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
def model(model_path, revision, npu_params):
    npu_params.warn_unused({"base"})
    model_kwargs = npu_params.base
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
        revision=revision
    )
    yield tokenizer


def test_cache(model, tokenizer):
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False)

    messages = [{"role": "user", "content": "My name is James."}]

    past_key_values = MobilintCache(model.get_cache_mxq_model())

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    )
    
    prefix_length = input_ids.shape[1]

    output_ids = model.generate(
        input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        do_sample=False,
        streamer=streamer,
        max_new_tokens=1024,
    )
    
    assistant_text = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
    messages += [{"role": "assistant", "content": assistant_text}]
    messages += [{"role": "user", "content": "What is my name?"}]

    past_key_values.dump_cache_memory()

    model.dispose()
    model.launch()

    past_key_values.layers[0]._seen_tokens = prefix_length
    past_key_values.load_cache_memory()
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    )
    
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False)
    
    output_ids = model.generate(
        input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        do_sample=False,
        streamer=streamer,
    )
    final_message = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
    assert "James" in final_message
