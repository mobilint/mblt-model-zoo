from typing import Any

from transformers import Union
from transformers.configuration_utils import SpecificPretrainedConfigType
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.aya_vision.configuration_aya_vision import AyaVisionConfig

from ...models.cohere2.configuration_cohere2 import MobilintCohere2Config

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintAyaVisionConfig(MobilintConfigMixin, AyaVisionConfig):
    model_type = "mobilint-aya_vision"
    sub_configs = {"text_config": MobilintCohere2Config, "vision_config": AutoConfig}

    def __init__(self, vision_config = None, text_config = None, **kwargs):        
        if text_config is None:
            text_config = {}
            text_config["model_type"] = "mobilint-cohere2"

        super().__init__(vision_config, text_config, **kwargs)

        if self.vision_feature_select_strategy != "full":
            raise ValueError(
                "vision_feature_select_strategy should be 'full'."
                f"Got: {self.vision_feature_select_strategy}"
            )
            
    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintAyaVisionConfig", tuple["MobilintAyaVisionConfig", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config: MobilintAyaVisionConfig
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs) # type: ignore
        
        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path
        
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

AutoConfig.register("mobilint-aya_vision", MobilintAyaVisionConfig)
