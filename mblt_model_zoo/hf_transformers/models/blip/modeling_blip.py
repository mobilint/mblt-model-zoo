from typing import cast

import torch
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.blip.modeling_blip import (
    BlipForConditionalGenerationModelOutput,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_blip import MobilintBlipConfig, MobilintBlipVisionConfig
from .modeling_blip_text import MobilintBlipTextLMHeadModel

logger = logging.get_logger(__name__)

class MobilintBlipPreTrainedModel(PreTrainedModel):
    config: MobilintBlipConfig
    base_model_prefix = "blip"
    input_modalities = ("image", "text")

class MobilintBlipVisionModel(MobilintModelMixin, MobilintBlipPreTrainedModel):
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    config: MobilintBlipVisionConfig

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if interpolate_pos_encoding is True:
            logger.warning("interpolate_pos_encoding is not supported.")
        
        last_hidden_state = self.mxq_forward(pixel_values).squeeze(2).permute((0, 2, 1))

        return BaseModelOutputWithPooling(
            last_hidden_state=cast(torch.FloatTensor, last_hidden_state),
            pooler_output=None,
        )

class MobilintBlipForConditionalGeneration(PretrainedOnlyMixin, MobilintGenerationMixin, MobilintBlipPreTrainedModel):
    base_model_prefix = "model"
    
    def __init__(self, config: MobilintBlipConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)
        
        self.vision_model = MobilintBlipVisionModel(config.vision_config, _internal_call=True)

        self.text_decoder = MobilintBlipTextLMHeadModel(config.text_config, _internal_call=True)

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

    def get_cache_mxq_model(self):
        return self.text_decoder.get_mxq_model()
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BlipForConditionalGenerationModelOutput:
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        image_embeds = vision_outputs.last_hidden_state

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            reduction="mean",
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[0]

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = cast(torch.LongTensor, (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            ))

        input_ids[:, 0] = self.config.text_config.bos_token_id if self.config.text_config.bos_token_id is not None else 0

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=None,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            **generate_kwargs,
        )

        return outputs

AutoModel.register(MobilintBlipConfig, MobilintBlipForConditionalGeneration)
AutoModelForVision2Seq.register(MobilintBlipConfig, MobilintBlipForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintBlipConfig, MobilintBlipForConditionalGeneration)
