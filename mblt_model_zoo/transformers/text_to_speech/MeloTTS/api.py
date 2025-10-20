from __future__ import annotations
from typing import TYPE_CHECKING
import noisereduce as nr
import torch
from torch import nn

from .models import MobilintSynthesizerTrn
from .download_utils import load_or_download_config, load_or_download_model

MISSING_MSG = (
    "Optional dependency 'melo' not found."
    "Please install MeloTTS (https://github.com/myshell-ai/MeloTTS)."
)

try:
    from melo.api import TTS as OriginalTTS
    import soundfile
except Exception:
    class OriginalTTS:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(MISSING_MSG)

        def __getattr__(self, name):
            raise ModuleNotFoundError(MISSING_MSG)

if TYPE_CHECKING:
    from melo.api import TTS as OriginalTTS
    import soundfile

class TTS(OriginalTTS):
    def __init__(self, 
                language,
                device='auto',
                use_hf=False,
                config_path=None,
                ckpt_path=None):
        nn.Module.__init__(self)
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = MobilintSynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        audio = super().tts_to_file(text, speaker_id, None, sdp_ratio, noise_scale, noise_scale_w, speed, pbar, format, position, quiet)

        audio = nr.reduce_noise(y=audio, sr=self.hps.data.sampling_rate, stationary=True, padding=0, 
                                prop_decrease=0.7,         # moderate noise reduction for better volume
                                freq_mask_smooth_hz=100,   # tight frequency mask for precise processing
                                time_mask_smooth_ms=150,   # shorter time mask for better volume preservation
                                n_fft=1024,                # standard FFT resolution for balanced processing
                                n_std_thresh_stationary=1.5,  # lower threshold for better volume preservation
                                clip_noise_stationary=True)    # clip noise for cleaner output
        
        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
