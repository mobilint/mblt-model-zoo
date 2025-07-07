from mblt_model_zoo.transformers import pipeline
from datasets import load_dataset

pipe = pipeline("automatic-speech-recognition", model="mobilint/whisper-small")

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0:2]["audio"]

print("model inference start")
outputs = pipe(sample)
print(outputs)
