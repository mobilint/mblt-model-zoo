from . import (
    cleaned_text_to_sequence,
    english,
    korean,
)

language_module_map = {"EN": english, 'KR': korean}


def clean_text(text, language, tokenizer=None):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text, tokenizer)
    return norm_text, phones, tones, word2ph


def text_to_sequence(text, language, tokenizer=None):
    norm_text, phones, tones, word2ph = clean_text(text, language, tokenizer)
    return cleaned_text_to_sequence(phones, tones, language)
