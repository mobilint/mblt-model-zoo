import sys

import torch

from .symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

models = {}
tokenizers = {}
def get_bert_feature(text, word2ph, device=None, model_id=''):
    from mblt_model_zoo.transformers import AutoModelForMaskedLM, AutoTokenizer
    global model
    global tokenizer
    
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model_id not in models:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(
            device
        )
        models[model_id] = model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizers[model_id] = tokenizer
    else:
        model = models[model_id]
        tokenizer = tokenizers[model_id]
        
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # bert models are compiled to output third-from-last hidden state
        res = model(**inputs)
        res = torch.cat(res["hidden_states"][0:1], -1)[0].cpu()
        
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

def get_bert(norm_text, word2ph, language, device, model_id):
    bert = get_bert_feature(norm_text, word2ph, device, model_id)
    return bert