import requests
from PIL import Image
from mblt_model_zoo.transformers import pipeline, AutoTokenizer
from transformers import TextStreamer

model_name = "mobilint/blip-image-captioning-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
    "image-text-to-text",
    model=model_name,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# conditional image captioning
text = "a photography of"
pipe(raw_image, text)

# unconditional image captioning
pipe(raw_image, "")
