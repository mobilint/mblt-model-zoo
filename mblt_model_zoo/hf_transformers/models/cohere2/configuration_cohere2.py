from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.cohere2.configuration_cohere2 import Cohere2Config

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintCohere2Config(MobilintConfigMixin, Cohere2Config):
    model_type = "mobilint-cohere2"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-cohere2", MobilintCohere2Config)
