from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.aya_vision.configuration_aya_vision import AyaVisionConfig

from ...models.cohere2.configuration_cohere2 import MobilintCohere2Config
from ...models.siglip.configuration_siglip import MobilintSiglipVisionConfig
from ...utils.configuration_utils import MobilintVisionTextConfigMixin


class MobilintAyaVisionConfig(MobilintVisionTextConfigMixin, AyaVisionConfig):
    model_type = "mobilint-aya_vision"
    sub_configs = {"text_config": MobilintCohere2Config, "vision_config": MobilintSiglipVisionConfig}

    def __init__(self, vision_config=None, text_config=None, **kwargs):
        if vision_config is None:
            vision_config = {"model_type": "mobilint-siglip_vision_model"}
        elif isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.setdefault("model_type", "mobilint-siglip_vision_model")

        if text_config is None:
            text_config = {"model_type": "mobilint-cohere2"}
        elif isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.setdefault("model_type", "mobilint-cohere2")

        AyaVisionConfig.__init__(self, vision_config=vision_config, text_config=text_config, **kwargs)

        if self.vision_feature_select_strategy != "full":
            raise ValueError(
                f"vision_feature_select_strategy should be 'full'. Got: {self.vision_feature_select_strategy}"
            )


AutoConfig.register("mobilint-aya_vision", MobilintAyaVisionConfig)
