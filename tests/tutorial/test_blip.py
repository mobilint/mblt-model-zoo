import requests
from PIL import Image
from mblt_model_zoo.transformers import pipeline

pipe = pipeline("image-text-to-text", model="mobilint/blip-image-captioning-large")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# conditional image captioning
text = "a photography of"
outputs = pipe(raw_image, text)
print(outputs)

# unconditional image captioning
outputs = pipe(raw_image, "")
print(outputs)
