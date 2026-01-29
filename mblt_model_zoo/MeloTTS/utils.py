import json
from abc import ABCMeta, abstractmethod

import torch

from . import commons
from .text import cleaned_text_to_sequence, get_bert
from .text.cleaner import clean_text


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParamsBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class TrainHParams(HParamsBase):
    def __init__(self, **kwargs):
        self.segment_size: int = kwargs['segment_size'] # type: ignore


class DataHParams(HParamsBase):
    def __init__(self, **kwargs):
        self.sampling_rate: int = kwargs['sampling_rate'] # type: ignore
        self.filter_length: int = kwargs['filter_length'] # type: ignore
        self.hop_length: int = kwargs['hop_length'] # type: ignore
        self.add_blank: bool = kwargs['add_blank'] # type: ignore
        self.n_speakers: int = kwargs.get('n_speakers', 1) # type: ignore
        self.spk2id: dict[str, int] = kwargs.get('spk2id', {}) # type: ignore


class ModelHParams(HParamsBase):
    def __init__(self, **kwargs):
        self.bert_model_id: str = kwargs['bert_model_id'] # type: ignore
        self.dev_no: int = kwargs.get('dev_no', 0)
        self.target_core: str = kwargs['target_core']
        self.encoder_mxq_path: str = kwargs['encoder_mxq_path'] # type: ignore
        self.decoder_mxq_path: str = kwargs['decoder_mxq_path'] # type: ignore
        self.use_spk_conditioned_encoder: bool = kwargs.get('use_spk_conditioned_encoder', True) # type: ignore
        self.use_noise_scaled_mask: bool = kwargs.get('use_noise_scaled_mask', True) # type: ignore
        self.use_mel_posterior_encoder: bool = kwargs.get('use_mel_posterior_encoder', False) # type: ignore
        self.use_duration_discriminator: bool = kwargs.get('use_duration_discriminator', True) # type: ignore
        self.inter_channels: int = kwargs.get('inter_channels', 192) # type: ignore
        self.hidden_channels: int = kwargs.get('hidden_channels', 192) # type: ignore
        self.filter_channels: int = kwargs.get('filter_channels', 768) # type: ignore
        self.n_heads: int = kwargs.get('n_heads', 2) # type: ignore
        self.n_layers: int = kwargs.get('n_layers', 6) # type: ignore
        self.n_layers_trans_flow: int = kwargs.get('n_layers_trans_flow', 3) # type: ignore
        self.kernel_size: int = kwargs.get('kernel_size', 3) # type: ignore
        self.p_dropout: float = kwargs.get('p_dropout', 0.1) # type: ignore
        self.resblock: int = kwargs.get('resblock', 1) # type: ignore
        self.resblock_kernel_sizes: list[int] = kwargs.get('resblock_kernel_sizes', [3, 7, 11]) # type: ignore
        self.resblock_dilation_sizes: list[list[int]] = kwargs.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]]) # type: ignore
        self.upsample_rates: list[int] = kwargs.get('upsample_rates', [8, 8, 2, 2, 2]) # type: ignore
        self.upsample_initial_channel: int = kwargs.get('upsample_initial_channel', 512) # type: ignore
        self.upsample_kernel_sizes: list[int] = kwargs.get('upsample_kernel_sizes', [16, 16, 8, 2, 2]) # type: ignore
        self.n_layers_q: int = kwargs.get('n_layers_q', 3) # type: ignore
        self.use_spectral_norm: bool = kwargs.get('use_spectral_norm', False) # type: ignore
        self.gin_channels: int = kwargs.get('gin_channels', 256) # type: ignore


class HParams(HParamsBase):
    def __init__(self, **kwargs):        
        self.train = TrainHParams(**kwargs['train'])
        self.data = DataHParams(**kwargs['data'])
        self.model = ModelHParams(**kwargs['model'])

        self.num_languages: int = kwargs.get('num_languages') # type: ignore
        self.num_tones: int = kwargs.get('num_tones') # type: ignore
        self.symbols: list[str] = kwargs.get('symbols') # type: ignore


def get_text_for_tts_infer(text, language_str, hps: HParams, device, symbol_to_id=None, tokenizer=None, bert=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str, tokenizer=tokenizer)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(norm_text, word2ph, device, tokenizer=tokenizer, bert=bert)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language
