from __future__ import annotations
import re
from tqdm import tqdm
import torch
from torch import nn
from typing import TYPE_CHECKING
import noisereduce as nr

MISSING_MSG = (
    "",
    "========================================================================================================",
    "Optional dependency 'melo' not found. Please install MeloTTS (https://github.com/myshell-ai/MeloTTS).",
    "NOTE: Default dependencies of MeloTTS contains old version of `transformers`, which is not compatible with our model zoo.",
    "You can modify `requirements.txt` in MeloTTS repository to remove `transformers` dependency",
    "========================================================================================================",
    "",
)

try:
    from melo.api import TTS as OriginalTTS
    import soundfile
    from . import utils
    from .models import MobilintSynthesizerTrn
    from .download_utils import load_or_download_config, load_or_download_model
except Exception:
    class OriginalTTS:
        def __init__(self, *args, **kwargs):
            for msg in MISSING_MSG:
                print(msg)
            raise ModuleNotFoundError()

        def __getattr__(self, name):
            for msg in MISSING_MSG:
                print(msg)
            raise ModuleNotFoundError()

if TYPE_CHECKING:
    from melo.api import TTS as OriginalTTS
    import soundfile
    from . import utils
    from .models import MobilintSynthesizerTrn
    from .download_utils import load_or_download_config, load_or_download_model

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
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

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
