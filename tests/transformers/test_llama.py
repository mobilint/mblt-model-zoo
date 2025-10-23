from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoTokenizer


def test_llama():
    model_path = "mobilint/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    )
    pipe.generation_config.max_new_tokens = None

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipe(
        messages,
        max_length=4096,
    )
