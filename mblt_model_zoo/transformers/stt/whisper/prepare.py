import os
from urllib.parse import urljoin
from ....utils.downloads import download_url_to_folder, download_url_to_file

WHISPER_URL = "https://dl.mobilint.com/model/aries/global/transformers/stt/whisper-small/"  # must endswith /

WHISPER_SMALL_FILE_LIST = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocab.json",
    "whisper-small_encoder.mxq",
    "whisper-small_decoder.mxq",
]


def prepare_files(
    dst: str = None,
):
    if dst is None:
        HOME_PATH = os.path.expanduser("~/.mblt_model_zoo/aries/global/whisper-small/")
        download_url_to_folder(WHISPER_URL, WHISPER_SMALL_FILE_LIST, HOME_PATH)
    else:
        os.makedirs(dst, exist_ok=True)
        for filename in WHISPER_SMALL_FILE_LIST:
            if not os.path.exists(os.path.join(dst, filename)):
                download_url_to_file(
                    urljoin(WHISPER_URL, filename), dst=os.path.join(dst, filename)
                )
