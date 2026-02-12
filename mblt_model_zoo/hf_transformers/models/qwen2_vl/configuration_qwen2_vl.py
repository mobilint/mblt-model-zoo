from typing import Any, Union

from transformers.configuration_utils import SpecificPretrainedConfigType
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLTextConfig,
    Qwen2VLVisionConfig,
)

from ...utils.configuration_utils import (
    MobilintConfigMixin,
    MobilintVisionTextConfigMixin,
)


class MobilintQwen2VLVisionConfig(MobilintConfigMixin, Qwen2VLVisionConfig):
    model_type = "mobilint-qwen2_vl"

class MobilintQwen2VLTextConfig(MobilintConfigMixin, Qwen2VLTextConfig):
    model_type = "mobilint-qwen2_vl_text"

class MobilintQwen2VLConfig(MobilintVisionTextConfigMixin, Qwen2VLConfig):
    model_type = "mobilint-qwen2_vl"
    sub_configs = {"vision_config": MobilintQwen2VLVisionConfig, "text_config": MobilintQwen2VLTextConfig}
    
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False
        self._attn_implementation = "eager"

AutoConfig.register("mobilint-qwen2_vl", MobilintQwen2VLConfig)
