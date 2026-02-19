from typing import Union, cast

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.blip.modeling_blip_text import BlipTextEmbeddings
from transformers.utils.generic import logging

from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_blip import MobilintBlipTextConfig

logger = logging.get_logger(__name__)

class MobilintBlipTextPreTrainedModel(PreTrainedModel):
    config: MobilintBlipTextConfig
    base_model_prefix = "bert"

class MobilintBlipTextModel(MobilintModelMixin, MobilintBlipTextPreTrainedModel):
    def __init__(self, config: MobilintBlipTextConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        self.embeddings = BlipTextEmbeddings(config)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings
    
    def forward(
        self,
        input_ids: Union[torch.Tensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.Tensor, None] = None,
        inputs_embeds: Union[torch.Tensor, None] = None,
        encoder_embeds: Union[torch.Tensor, None] = None,
        encoder_hidden_states: Union[torch.Tensor, None] = None,
        encoder_attention_mask: Union[torch.Tensor, None] = None,
        past_key_values: Union[MobilintCache, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        is_decoder: Union[bool, None] = False,
        cache_position: Union[torch.Tensor, None] = None,
        **kwargs,
    ) -> Union[tuple[Union[torch.Tensor, None, MobilintCache], ...], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")
        
        assert encoder_hidden_states is not None, "encoder_hidden_states is None!"
        
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        embedding_output: torch.Tensor = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(torch.LongTensor, torch.arange(
                past_seen_tokens, past_seen_tokens + embedding_output.shape[1], device=embedding_output.device
            ))
        
        logits = self.decoder_forward(
            embedding_output.unsqueeze(1),
            encoder_hidden_states.unsqueeze(1),
            past_key_values,
            cache_position,
        ).squeeze(0)

        if not return_dict:
            return (logits, None, past_key_values)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=cast(torch.FloatTensor, logits),
            pooler_output=None,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

class MobilintBlipTextLMHeadModel(MobilintBlipTextPreTrainedModel, MobilintGenerationMixin):
    def __init__(self, config: MobilintBlipTextConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        self.bert = MobilintBlipTextModel(config, add_pooling_layer=False, _internal_call=True)
        self.label_smoothing = config.label_smoothing
    
    def get_cache_mxq_model(self):
        return self.bert.get_mxq_model()

    def forward(
        self,
        input_ids: Union[torch.Tensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.Tensor, None] = None,
        inputs_embeds: Union[torch.Tensor, None] = None,
        encoder_hidden_states: Union[torch.Tensor, None] = None,
        encoder_attention_mask: Union[torch.Tensor, None] = None,
        labels: Union[torch.Tensor, None] = None,
        past_key_values: Union[MobilintCache, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        return_logits: Union[bool, None] = False,
        is_decoder: Union[bool, None] = True,
        reduction: Union[str, None] = "mean",
        cache_position: Union[torch.Tensor, None] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[Union[torch.Tensor, MobilintCache], ...], CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
            
        if logits_to_keep > 1:
            logger.warning("logits_to_keep larger than 1 is not supported: %d" % logits_to_keep)
        
        if reduction != "mean":
            logger.warning("reduction except 'mean' is not supported: %s" % reduction)
            reduction = "mean"
        
        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            cache_position=cache_position,
        )
        
        logits = outputs[0]
        
        if return_logits:
            return logits

        lm_loss: Union[torch.Tensor, None] = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = logits
            labels = labels[:, 1:].contiguous().to(shifted_prediction_scores.device)
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=self.label_smoothing)
            lm_loss = cast(torch.Tensor, loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)))

        if not return_dict:
            output = (logits, past_key_values) if past_key_values is not None else (logits,)
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=cast(torch.FloatTensor, lm_loss),
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # Overwrite -- hardcoded key return (`is_decoder=True`)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        model_inputs["is_decoder"] = True

        return model_inputs
    
