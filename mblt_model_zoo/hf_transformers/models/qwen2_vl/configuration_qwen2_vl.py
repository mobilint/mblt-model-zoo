from typing import Any, Union

from transformers.configuration_utils import SpecificPretrainedConfigType
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLTextConfig,
    Qwen2VLVisionConfig,
)

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintQwen2VLVisionConfig(MobilintConfigMixin, Qwen2VLVisionConfig):
    model_type = "mobilint-qwen2_vl"

class MobilintQwen2VLTextConfig(MobilintConfigMixin, Qwen2VLTextConfig):
    model_type = "mobilint-qwen2_vl_text"

class MobilintQwen2VLConfig(Qwen2VLConfig):
    model_type = "mobilint-qwen2_vl"
    sub_configs = {"vision_config": MobilintQwen2VLVisionConfig, "text_config": MobilintQwen2VLTextConfig}
    
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False
        self._attn_implementation = "eager"

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintQwen2VLConfig", tuple["MobilintQwen2VLConfig", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config: MobilintQwen2VLConfig
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs) # type: ignore
        
        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path
        
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

AutoConfig.register("mobilint-qwen2_vl", MobilintQwen2VLConfig)
