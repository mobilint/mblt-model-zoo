from mblt_model_zoo.transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="mobilint/aya-vision-8b",
)

# Format message with the aya-vision chat template
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Which one is shown in this picture?"},
    ]
}]

outputs = pipe(
    text=messages,
    max_new_tokens=300,
    return_full_text=False,
)

print(outputs)