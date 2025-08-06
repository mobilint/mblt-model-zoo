from mblt_model_zoo.transformers import pipeline, AutoTokenizer
from transformers import TextStreamer

model_name = "mobilint/aya-vision-8b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
    "image-text-to-text",
    model=model_name,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    use_fast=True,
)

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
    padding='max_length',
    truncation=True,
    max_length=4096,
)