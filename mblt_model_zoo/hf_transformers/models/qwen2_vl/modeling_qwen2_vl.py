from typing import cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForImageTextToText,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen2_vl import (
    MobilintQwen2VLConfig,
    MobilintQwen2VLTextConfig,
    MobilintQwen2VLVisionConfig,
)

logger = logging.get_logger(__name__)

class MobilintQwen2VLPreTrainedModel(PreTrainedModel):
    config: MobilintQwen2VLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")

class MobilintQwen2VisionTransformerPretrainedModel(MobilintModelMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLVisionConfig
    input_modalities = ("image", "video")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        gt, gh, gw = grid_thw[0].tolist()

        c  = 3
        pt = 2
        Mh = 2
        Mw = 2
        gh2 = gh // 2
        gw2 = gw // 2
        ph = pw = int((hidden_states.shape[-1] // (pt * c)) ** 0.5)

        assert hidden_states.shape[0] == gt * gh2 * gw2 * Mh * Mw
        assert hidden_states.shape[1] == c * pt * ph * pw

        # (gt, gh2, gw2, Mh, Mw, c, pt, ph, pw)
        hidden_states = hidden_states.view(gt, gh2, gw2, Mh, Mw, c, pt, ph, pw)

        # rearrange: "(gt gh gw Mh Mw) (c pt ph pw) -> gt (gh gw ph) (Mh Mw pw) (pt c)"
        # (gt, pt, c, Mh, Mw, pw, gh2, gw2, ph)
        hidden_states = hidden_states.permute(0, 1, 2, 7, 3, 4, 8, 6, 5).contiguous()

        # gt (gh gw ph) (Mh Mw pw) (pt c)
        hidden_states = hidden_states.view(
            gt,
            gh2 * gw2 * ph,
            Mh * Mw * pw,
            pt * c,
        ).squeeze(0)
                
        merged_hidden_states = self.mxq_forward(hidden_states)

        return merged_hidden_states

class MobilintQwen2VLTextModel(MobilintModelMixin, MobilintGenerationMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLTextConfig
    input_modalities = ("text",)
    
    def __init__(self, config: MobilintQwen2VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: MobilintCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        chunk_size: int = 0,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        assert inputs_embeds is not None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(torch.LongTensor, torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ))
        
        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")
        
        logits = self.llm_forward(inputs_embeds, past_key_values, cache_position, chunk_size)

        if not return_dict:
            return tuple(
                v for v in [logits, past_key_values, None, None] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

class MobilintQwen2VLModel(PretrainedOnlyMixin, MobilintQwen2VLPreTrainedModel, Qwen2VLModel):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        MobilintQwen2VLPreTrainedModel.__init__(self, config, *args, **kwargs)
        self.visual = MobilintQwen2VisionTransformerPretrainedModel._from_config(config.vision_config, _internal_call=True)
        self.language_model = MobilintQwen2VLTextModel._from_config(config.text_config, _internal_call=True)
        self.rope_deltas = None  # cache rope_deltas here

class MobilintQwen2VLForConditionalGeneration(PretrainedOnlyMixin, MobilintQwen2VLPreTrainedModel, MobilintGenerationMixin, Qwen2VLForConditionalGeneration):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)
        
        self.model = MobilintQwen2VLModel(config, _internal_call=True)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()
    
    def get_cache_mxq_model(self):
        return self.model.language_model.get_mxq_model()
        
AutoModel.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
