from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.siglip.configuration_siglip import SiglipConfig
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.modeling_utils import MobilintModelMixin
from .configuration_siglip import MobilintSiglipVisionConfig

logger = logging.get_logger(__name__)

class MobilintSiglipPreTrainedModel(PreTrainedModel):
    config: SiglipConfig
    base_model_prefix = "siglip"
    input_modalities = ("image", "text")

class MobilintSiglipVisionModel(MobilintModelMixin, MobilintSiglipPreTrainedModel):
    config: MobilintSiglipVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        image_features = self.mxq_forward(pixel_values).permute(0, 2, 3, 1).contiguous()
        return image_features

AutoModel.register(MobilintSiglipVisionConfig, MobilintSiglipVisionModel)
