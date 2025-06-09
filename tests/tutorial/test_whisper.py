import mblt_model_zoo.stt.whisper
from mblt_model_zoo.stt.whisper.prepare import prepare_files
from transformers import pipeline
from datasets import load_dataset
import os


REPO_ID = "openai/whisper-small" # whisper-small model
prepare_files(REPO_ID) # the files will be saved in ~/.mblt_model_zoo/

HOME_PATH = os.path.expanduser("~")
MODEL_NAME = REPO_ID.strip("/").split("/")[-1]
MODEL_PATH = f"{HOME_PATH}/.mblt_model_zoo/{MODEL_NAME}"

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

print('model inference start')
outputs = pipe(sample)
print(outputs)