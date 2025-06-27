import os
from mblt_model_zoo.transformers.stt.whisper.prepare import prepare_files
from transformers import pipeline
from datasets import load_dataset

MODEL_NAME = "whisper-small"
HOME_PATH = os.path.expanduser("~")
MODEL_PATH = f"{HOME_PATH}/.mblt_model_zoo/{MODEL_NAME}"

prepare_files(MODEL_PATH)

pipe = pipeline("automatic-speech-recognition", model=MODEL_PATH)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0:2]["audio"]

print("model inference start")
outputs = pipe(sample)
print(outputs)
