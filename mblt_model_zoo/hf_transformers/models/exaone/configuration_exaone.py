from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

from ...utils.configuration_utils import MobilintConfigMixin
from .original.configuration_exaone import ExaoneConfig


class MobilintExaoneConfig(MobilintConfigMixin, ExaoneConfig):
    model_type = "mobilint-exaone"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-exaone", MobilintExaoneConfig)
AutoTokenizer.register(MobilintExaoneConfig, slow_tokenizer_class=GPT2Tokenizer, fast_tokenizer_class=GPT2TokenizerFast)
