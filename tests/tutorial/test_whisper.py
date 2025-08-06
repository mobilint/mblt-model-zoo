from mblt_model_zoo.transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from transformers import TextStreamer

model_path = "mobilint/whisper-small"

tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]

prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
print(prediction)
