from melo.text import get_bert as original_get_bert

def get_bert(norm_text, word2ph, language, device):
    from .english_bert import get_bert_feature as en_bert

    lang_bert_func_map = {"EN": en_bert}
    
    if language:
      bert = lang_bert_func_map[language](norm_text, word2ph, device)
    else:
      bert = original_get_bert(norm_text, word2ph, language, device)
    return bert
