import os
import torch
import melo.utils
from ...utils.auto import convert_identifier_to_path

LANG_TO_HF_REPO_ID = {
    'EN_NEWEST': 'mobilint/MeloTTS-English-v3',
}

def load_or_download_config(locale, use_hf=False, config_path=None):
    assert use_hf is False
    
    if config_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        download_path = convert_identifier_to_path(LANG_TO_HF_REPO_ID[language])
        config_path = os.path.join(download_path, "config.json")
    
    result = melo.utils.get_hparams_from_file(config_path)
    result.model.mxq_path_enc_p_sdp_dp = os.path.join(os.path.dirname(config_path), result.model.mxq_path_enc_p_sdp_dp)
    result.model.mxq_path_dec_flow = os.path.join(os.path.dirname(config_path), result.model.mxq_path_dec_flow)
    
    return result

def load_or_download_model(locale, device, use_hf=False, ckpt_path=None):
    assert use_hf is False
    
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        download_path = convert_identifier_to_path(LANG_TO_HF_REPO_ID[language])
        ckpt_path = os.path.join(download_path, "checkpoint.pth")
    return torch.load(ckpt_path, map_location=device)

from ...utils.types import TransformersModelInfo

EN_NEWEST = TransformersModelInfo(
    original_model_id="myshell-ai/MeloTTS-English-v3",
    model_id="mobilint/MeloTTS-English-v3",
    download_url_base="https://dl.mobilint.com/model/transformers/tts/MeloTTS-English-v3/",
    file_list=[
        "checkpoint.pth",
        "config.json",
        "MeloTTS-English-v3_enc_p_sdp_dp.mxq",
        "MeloTTS-English-v3_dec_flow.mxq",
    ],
)
