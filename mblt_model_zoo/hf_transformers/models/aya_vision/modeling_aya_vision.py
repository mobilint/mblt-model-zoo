from typing import Optional, Union, cast

import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...models.cohere2.modeling_cohere2 import MobilintCohere2ForCausalLM
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_aya_vision import MobilintAyaVisionConfig

logger = logging.get_logger(__name__)

class MobilintAyaVisionForCausalLM(MobilintModelMixin, MobilintGenerationMixin):
    config: MobilintAyaVisionConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")

    def __init__(self, config: MobilintAyaVisionConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.language_model: MobilintCohere2ForCausalLM = AutoModelForCausalLM.from_config(config.text_config, _internal_call=True)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        image_features = self.mxq_forward(pixel_values).permute(0, 2, 3, 1).contiguous()
        return image_features

    def get_placeholder_mask(
        self, input_ids: Optional[torch.LongTensor], inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        assert inputs_embeds[special_image_mask].numel() == image_features.numel(), (
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask
    
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: MobilintCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        chunk_size: int = 128,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AyaVisionCausalLMOutputWithPast:
        if logits_to_keep > 1:
            logger.warning("logits_to_keep larger than 1 is not supported: %d" % logits_to_keep)
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        assert inputs_embeds is not None

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=cast(torch.FloatTensor, image_features)
            )
            inputs_embeds = cast(torch.FloatTensor, inputs_embeds.masked_scatter(special_image_mask, image_features))

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=kwargs.pop('use_cache', None),
            cache_position=cache_position,
            image_sizes=image_sizes,
            chunk_size=chunk_size,
            **kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return AyaVisionCausalLMOutputWithPast(
            loss=loss,
            logits=cast(torch.FloatTensor, logits),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=cast(torch.FloatTensor | None, image_features if pixel_values is not None else None),
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = MobilintGenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if (cache_position is not None and cache_position[0] == 0 or
            is_first_iteration or not kwargs.get("use_cache", True)):
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

        
AutoModel.register(MobilintAyaVisionConfig, MobilintAyaVisionForCausalLM)
AutoModelForImageTextToText.register(MobilintAyaVisionConfig, MobilintAyaVisionForCausalLM)
