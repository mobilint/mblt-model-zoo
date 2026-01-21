from typing import Optional, Union, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_llama import MobilintLlamaConfig

logger = logging.get_logger(__name__)

class MobilintLlamaForCausalLM(MobilintModelMixin, MobilintGenerationMixin):
    config: MobilintLlamaConfig

    def __init__(self, config: MobilintLlamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: int = 128,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        logits = self.llm_forward(inputs_embeds, past_key_values, cache_position, chunk_size)

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
        
AutoModel.register(MobilintLlamaConfig, MobilintLlamaForCausalLM)
AutoModelForCausalLM.register(MobilintLlamaConfig, MobilintLlamaForCausalLM)
