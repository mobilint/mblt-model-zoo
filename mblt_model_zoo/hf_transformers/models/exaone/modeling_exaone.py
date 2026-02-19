from typing import Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.utils.generic import logging

from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_exaone import MobilintExaoneConfig

logger = logging.get_logger(__name__)

class MobilintExaoneForCausalLM(MobilintModelMixin, MobilintGenerationMixin):
    config: MobilintExaoneConfig
    base_model_prefix = "transformer"

    def __init__(self, config: MobilintExaoneConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.config.pad_token_id)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        chunk_size: int = 128,
        count_npu_time: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, MobilintCache], ...], CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        assert inputs_embeds is not None
        
        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(torch.LongTensor, torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ))
        
        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")

        lm_logits = self.llm_forward(
            inputs_embeds,
            past_key_values,
            cache_position,
            chunk_size,
            count_npu_time=count_npu_time,
        )

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(self.wte.weight.dtype)
            loss = loss.to(self.wte.weight.dtype)

        if not return_dict:
            model_output: tuple[MobilintCache] = tuple(v for v in [None, past_key_values, None, None] if v is not None)
            output = (lm_logits,) + model_output
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=cast(torch.FloatTensor, lm_logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

        
AutoModel.register(MobilintExaoneConfig, MobilintExaoneForCausalLM)
AutoModelForCausalLM.register(MobilintExaoneConfig, MobilintExaoneForCausalLM)
