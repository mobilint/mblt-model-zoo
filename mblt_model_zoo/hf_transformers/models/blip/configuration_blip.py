from typing import Any, Union

from transformers.configuration_utils import (
    PretrainedConfig,
    SpecificPretrainedConfigType,
)
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.blip.configuration_blip import (
    BlipConfig,
    BlipTextConfig,
    BlipVisionConfig,
)
from transformers.utils import logging

from ...utils.configuration_utils import MobilintConfigMixin

logger = logging.get_logger(__name__)

class MobilintBlipVisionConfig(MobilintConfigMixin, BlipVisionConfig):
    model_type = "mobilint-blip_vision_model"

class MobilintBlipTextConfig(MobilintConfigMixin, BlipTextConfig):
    model_type = "mobilint-blip_text_model"
    
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

class MobilintBlipConfig(BlipConfig):
    model_type = "mobilint-blip"
    sub_configs = {"vision_config": MobilintBlipVisionConfig, "text_config": MobilintBlipTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        **kwargs,
    ):
        PretrainedConfig.__init__(self, **kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `MobilintBlipTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `MobilintBlipVisionConfig` with default values.")

        self.text_config = MobilintBlipTextConfig(**text_config)
        self.vision_config = MobilintBlipVisionConfig(**vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MobilintBlipTextConfig,
        vision_config: MobilintBlipVisionConfig,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintBlipConfig", tuple["MobilintBlipConfig", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config: MobilintBlipConfig
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs) # type: ignore
        
        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path
        
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

AutoConfig.register("mobilint-blip", MobilintBlipConfig)
