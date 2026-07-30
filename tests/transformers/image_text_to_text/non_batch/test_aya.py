from transformers import TextStreamer

MODEL_PATHS = ("mobilint/aya-vision-8b",)


def test_aya(pipe, generation_token_limit: int):
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    # Format message with the aya-vision chat template
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo=",
                },
                {"type": "text", "text": "Which one is shown in this picture?"},
            ],
        }
    ]

    pipe(
        text=messages,
        generate_kwargs={
            "max_new_tokens": generation_token_limit,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        },
    )
