from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintLlamaConfig(MobilintConfigMixin, LlamaConfig):
    model_type = "mobilint-llama"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-llama", MobilintLlamaConfig)
AutoTokenizer.register(MobilintLlamaConfig, slow_tokenizer_class=LlamaTokenizer, fast_tokenizer_class=LlamaTokenizerFast)
