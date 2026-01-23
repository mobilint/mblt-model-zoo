import os

import torch

from ...utils.auto import convert_identifier_to_path
from . import utils

LANG_TO_HF_REPO_ID = {
    'EN_NEWEST': 'mobilint/MeloTTS-English-v3',
    'KR': 'mobilint/MeloTTS-Korean',
}


def load_or_download_config(locale, use_hf=False, config_path=None):
    assert use_hf is False
    
    if config_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        download_path = convert_identifier_to_path(LANG_TO_HF_REPO_ID[language])
        config_path = os.path.join(download_path, "config.json")
    
    result = utils.get_hparams_from_file(config_path)
    result.model.mxq_path_enc_p_sdp_dp = os.path.join(os.path.dirname(config_path), result.model.mxq_path_enc_p_sdp_dp)
    result.model.mxq_path_dec_flow = os.path.join(os.path.dirname(config_path), result.model.mxq_path_dec_flow)
    
    # bert_model_id can be either model id such as `mobilint/bert-base-uncased` or path such as `bert-kor-base`
    bert_model_path = os.path.join(os.path.dirname(config_path), result.model.bert_model_id)
    if os.path.exists(bert_model_path):
        result.model.bert_model_id = bert_model_path
    
    return result


def load_or_download_model(locale, device, use_hf=False, ckpt_path=None):
    assert use_hf is False
    
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        download_path = convert_identifier_to_path(LANG_TO_HF_REPO_ID[language])
        ckpt_path = os.path.join(download_path, "checkpoint.pth")
    return torch.load(ckpt_path, map_location=device)
