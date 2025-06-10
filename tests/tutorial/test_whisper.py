from mblt_model_zoo.stt.whisper.prepare import prepare_files
from transformers import pipeline
from datasets import load_dataset
import os


REPO_ID = "openai/whisper-small"  # whisper-small model
local_path = "/workspace/mblt-model-zoo/tmp"
prepare_files(
    REPO_ID, local_path=local_path
)  # the files will be saved in /workspace/mblt-model-zoo/tmp

MODEL_NAME = REPO_ID.strip("/").split("/")[-1]
if local_path is not None:
    MODEL_PATH = os.path.join(local_path, MODEL_NAME)
    os.makedirs(MODEL_PATH, exist_ok=True)
else:
    HOME_PATH = os.path.expanduser("~")
    MODEL_PATH = f"{HOME_PATH}/.mblt_model_zoo/{MODEL_NAME}"

pipe = pipeline("automatic-speech-recognition", model=MODEL_PATH)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]

print("model inference start")
outputs = pipe(sample)
print(outputs)
