from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoProcessor

model_name = "mobilint/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
pipe = pipeline(
    "image-text-to-text",
    model=model_name,
    processor=processor,
    device="cpu",
)
pipe.generation_config.max_new_tokens = None

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
        "max_length": 4096,
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)
