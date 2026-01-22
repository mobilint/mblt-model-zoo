from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.bert.configuration_bert import BertConfig

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintBertConfig(MobilintConfigMixin, BertConfig):
    model_type = "mobilint-bert"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-bert", MobilintBertConfig)
