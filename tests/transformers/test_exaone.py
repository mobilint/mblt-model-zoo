from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


def test_exaone():
    model_name = "mobilint/EXAONE-3.5-2.4B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    )
    pipe.generation_config.max_new_tokens = None

    # Choose your prompt
    prompt = "Explain how wonderful you are"  # English example
    prompt = "스스로를 자랑해 봐"  # Korean example

    messages = [
        {
            "role": "system",
            "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(
        messages,
        max_length=4096,
    )
