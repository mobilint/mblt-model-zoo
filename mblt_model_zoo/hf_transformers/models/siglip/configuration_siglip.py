from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.siglip.configuration_siglip import (
    SiglipVisionConfig,
)
from transformers.utils import logging

from ...utils.configuration_utils import MobilintConfigMixin

logger = logging.get_logger(__name__)

class MobilintSiglipVisionConfig(MobilintConfigMixin, SiglipVisionConfig):
    model_type = "mobilint-siglip_vision_model"

AutoConfig.register("mobilint-siglip_vision_model", MobilintSiglipVisionConfig)
