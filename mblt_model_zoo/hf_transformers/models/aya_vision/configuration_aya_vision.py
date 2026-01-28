from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.aya_vision.configuration_aya_vision import AyaVisionConfig

from ...models.cohere2.configuration_cohere2 import MobilintCohere2Config
from ...models.siglip.configuration_siglip import MobilintSiglipVisionConfig

from ...utils.configuration_utils import MobilintVisionTextConfigMixin


class MobilintAyaVisionConfig(MobilintVisionTextConfigMixin, AyaVisionConfig):
    model_type = "mobilint-aya_vision"
    sub_configs = {"text_config": MobilintCohere2Config, "vision_config": MobilintSiglipVisionConfig}

    def __init__(self, vision_config = None, text_config = None, **kwargs):
        if vision_config is None:
            vision_config = {}
            vision_config["model_type"] = "mobilint-siglip_vision_model"

        if text_config is None:
            text_config = {}
            text_config["model_type"] = "mobilint-cohere2"

        super().__init__(vision_config, text_config, **kwargs)

        if self.vision_feature_select_strategy != "full":
            raise ValueError(
                "vision_feature_select_strategy should be 'full'."
                f"Got: {self.vision_feature_select_strategy}"
            )

AutoConfig.register("mobilint-aya_vision", MobilintAyaVisionConfig)
