from typing import Union, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_cohere2 import MobilintCohere2Config

logger = logging.get_logger(__name__)

class MobilintCohere2ForCausalLM(MobilintModelMixin, MobilintGenerationMixin):
    config: MobilintCohere2Config
    base_model_prefix = "model"

    def __init__(self, config: MobilintCohere2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.logit_scale = config.logit_scale
    
    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.LongTensor, None] = None,
        past_key_values: Union[MobilintCache, None] = None,
        inputs_embeds: Union[torch.FloatTensor, None] = None,
        labels: Union[torch.LongTensor, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: int = 128,
        count_npu_time: bool = False,
        **kwargs: Unpack[TransformersKwargs], # type: ignore
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if logits_to_keep > 1:
            logger.warning("logits_to_keep larger than 1 is not supported: %d" % logits_to_keep)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
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

        logits = self.llm_forward(
            inputs_embeds,
            past_key_values,
            cache_position,
            chunk_size,
            count_npu_time=count_npu_time,
        )
        logits = logits * self.logit_scale  # main diff from Llama

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        
AutoModel.register(MobilintCohere2Config, MobilintCohere2ForCausalLM)
AutoModelForCausalLM.register(MobilintCohere2Config, MobilintCohere2ForCausalLM)
