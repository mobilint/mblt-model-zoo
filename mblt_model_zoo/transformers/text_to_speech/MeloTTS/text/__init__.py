import torch
from mblt_model_zoo.transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

models = {}
tokenizers = {}

def get_bert_feature(text, word2ph, device=None, model_id=''):
    global model
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
        # just use last hidden state instead of third-from-last hidden state
        res = model(**inputs)[0].cpu()
        
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