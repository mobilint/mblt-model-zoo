import torch
from huggingface_hub import hf_hub_download

from .utils import get_hparams_from_file

LANG_TO_HF_REPO_ID = {
    'EN_NEWEST': 'mobilint/MeloTTS-English-v3',
    'KR': 'mobilint/MeloTTS-Korean',
}

def load_or_download_config(locale, config_path=None, local_files_only=False):
    if config_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        config_path = hf_hub_download(
            repo_id=LANG_TO_HF_REPO_ID[language],
            filename="config.json",
            local_files_only=local_files_only
        )
    return get_hparams_from_file(config_path)

def load_or_download_model(locale, device, ckpt_path=None, local_files_only=False):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        ckpt_path = hf_hub_download(
            repo_id=LANG_TO_HF_REPO_ID[language],
            filename="checkpoint.pth",
            local_files_only=local_files_only
        )
    return torch.load(ckpt_path, map_location=device)
