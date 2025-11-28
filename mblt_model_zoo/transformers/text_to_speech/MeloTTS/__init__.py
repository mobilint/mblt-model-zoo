from ...utils.types import TransformersModelInfo
from .api import TTS

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

KR = TransformersModelInfo(
    original_model_id="myshell-ai/MeloTTS-Korean",
    model_id="mobilint/MeloTTS-Korean",
    download_url_base="https://dl.mobilint.com/model/transformers/tts/MeloTTS-Korean/",
    file_list=[
        "checkpoint.pth",
        "config.json",
        "MeloTTS-Korean_enc_p_sdp_dp.mxq",
        "MeloTTS-Korean_dec_flow.mxq",
    ],
)