from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintQwen3Config(MobilintConfigMixin, Qwen3Config):
    model_type = "mobilint-qwen3"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-qwen3", MobilintQwen3Config)
