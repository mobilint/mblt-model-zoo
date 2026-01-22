from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintQwen2Config(MobilintConfigMixin, Qwen2Config):
    model_type = "mobilint-qwen2"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-qwen2", MobilintQwen2Config)
