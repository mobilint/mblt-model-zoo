from mblt_model_zoo.transformers import pipeline
from datasets import load_dataset
from transformers import TextStreamer

model_path = "mobilint/whisper-small"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_path,
)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]

output = pipe(
    sample.copy(),
    batch_size=8,
    return_timestamps=True,
    generate_kwargs={
        "max_length": 4096,
    },
)

print(output)