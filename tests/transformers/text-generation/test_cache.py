import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline

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
def pipe(request, revision, npu_params):
    model_path = request.param
    npu_params.warn_unused({"base"})
    model_kwargs = npu_params.base
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    if model_kwargs:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
        )
    yield pipe
    del pipe


def test_cache(pipe):
    pipe.generation_config.max_new_tokens = None

    messages = [{"role": "user", "content": "My name is James."}]

    past_key_values = MobilintCache(pipe.model.get_cache_mxq_model())

    outputs = pipe(
        messages,
        max_length=512,
        use_cache=True,
        past_key_values=past_key_values,
    )

    messages = outputs[0]['generated_text']
    messages += [{"role": "user", "content": "What is my name?"}]

    past_key_values.dump_cache_memory()

    pipe.model.dispose()
    pipe.model.launch()

    past_key_values.load_cache_memory()

    outputs = pipe(
        messages,
        max_length=512,
        use_cache=True,
        past_key_values=past_key_values,
    )