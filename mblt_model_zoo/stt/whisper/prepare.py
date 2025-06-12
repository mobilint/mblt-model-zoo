from huggingface_hub import snapshot_download
import os
from ...vision.utils.downloads import download_url_to_file


def prepare_files(
    REPO_ID="openai/whisper-small",
    base_dir="https://dl.mobilint.com/model/stt/whisper-small",
    encoder_mxq="whisper-small_encoder.mxq",
    decoder_mxq="whisper-small_decoder.mxq",
    model_cfg="config.json",
    local_path: str = None,
):
    MODEL_NAME = REPO_ID.strip("/").split("/")[-1]
    if local_path is not None:
        MODEL_PATH = os.path.join(local_path, MODEL_NAME)
        os.makedirs(MODEL_PATH, exist_ok=True)
    else:
        HOME_PATH = os.path.expanduser("~")
        MODEL_PATH = f"{HOME_PATH}/.mblt_model_zoo/{MODEL_NAME}"

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_PATH,
        ignore_patterns=[
            "pytorch_model.bin",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ],
    )

    download_url_to_file(
        url=f"{base_dir}/{encoder_mxq}", dst=f"{MODEL_PATH}/{encoder_mxq}"
    )
    download_url_to_file(
        url=f"{base_dir}/{decoder_mxq}", dst=f"{MODEL_PATH}/{decoder_mxq}"
    )
    download_url_to_file(url=f"{base_dir}/config.json", dst=f"{MODEL_PATH}/{model_cfg}")
