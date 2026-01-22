from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintQwen2Config(MobilintConfigMixin, Qwen2Config):
    model_type = "mobilint-qwen2"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-qwen2", MobilintQwen2Config)
AutoTokenizer.register(MobilintQwen2Config, slow_tokenizer_class=Qwen2Tokenizer, fast_tokenizer_class=Qwen2TokenizerFast)
