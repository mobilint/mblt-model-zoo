from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintLlamaConfig(MobilintConfigMixin, LlamaConfig):
    model_type = "mobilint-llama"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-llama", MobilintLlamaConfig)
