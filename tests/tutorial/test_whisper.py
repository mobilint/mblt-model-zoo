from mblt_model_zoo.transformers import pipeline
from datasets import load_dataset

pipe = pipeline("automatic-speech-recognition", model="mobilint/whisper-small")

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]

prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
print(prediction)
