from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.exaone4.configuration_exaone4 import Exaone4Config

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintExaone4Config(MobilintConfigMixin, Exaone4Config):
    model_type = "mobilint-exaone4"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-exaone4", MobilintExaone4Config)
