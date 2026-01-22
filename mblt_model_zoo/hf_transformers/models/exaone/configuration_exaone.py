from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import MobilintConfigMixin
from .original.configuration_exaone import ExaoneConfig


class MobilintExaoneConfig(MobilintConfigMixin, ExaoneConfig):
    model_type = "mobilint-exaone"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-exaone", MobilintExaoneConfig)
