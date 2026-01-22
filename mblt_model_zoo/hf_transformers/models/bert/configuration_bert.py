from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintBertConfig(MobilintConfigMixin, BertConfig):
    model_type = "mobilint-bert"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-bert", MobilintBertConfig)
AutoTokenizer.register(MobilintBertConfig, slow_tokenizer_class=BertTokenizer, fast_tokenizer_class=BertTokenizerFast)
