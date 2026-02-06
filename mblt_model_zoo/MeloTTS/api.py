import re
from typing import Optional

import numpy as np
import soundfile
import torch
from torch import nn
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from . import utils
from .download_utils import (
    LANG_TO_HF_REPO_ID,
    load_or_download_config,
    load_or_download_model,
)
from .models import MobilintSynthesizerTrn
from .split_utils import split_sentence


class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                config_path=None,
                ckpt_path=None,

                trust_remote_code: Optional[bool]=None,
                local_files_only: Optional[bool]=None,
                
                dev_no: Optional[int] = None,
                target_core: Optional[str] = None,
                encoder_mxq_path: Optional[str] = None,
                decoder_mxq_path: Optional[str] = None,
        ):
        nn.Module.__init__(self)
        if device == "auto":
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            if torch.backends.mps.is_available():
                device = "mps"
        if "cuda" in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, config_path=config_path, local_files_only=local_files_only)
        
        if dev_no is not None:
            hps.model.dev_no = dev_no
        
        if target_core is not None:
            hps.model.target_core = target_core
        
        if encoder_mxq_path is not None:
            hps.model.encoder_mxq_path = encoder_mxq_path
        
        if decoder_mxq_path is not None:
            hps.model.decoder_mxq_path = decoder_mxq_path

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
            name_or_path=LANG_TO_HF_REPO_ID[language],
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device

        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, ckpt_path=ckpt_path, local_files_only=local_files_only)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

        self.tokenizer = AutoTokenizer.from_pretrained(
            hps.model.bert_model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        self.bert = AutoModelForMaskedLM.from_pretrained(
            hps.model.bert_model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,

            dev_no=hps.model.dev_no,
            target_cores=[hps.model.target_core],
        )
    
    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.0):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print("\n".join(texts))
            print(" > ===========================")
        return texts
    
    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
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
            if language in ["EN", "ZH_MIX_EN"]:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id, tokenizer=self.tokenizer, bert=self.bert)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
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
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        
        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(
                    output_path, audio, self.hps.data.sampling_rate, format=format
                )
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)

    def launch(self):
        self.model.launch()

    def dispose(self):
        self.model.dispose()
