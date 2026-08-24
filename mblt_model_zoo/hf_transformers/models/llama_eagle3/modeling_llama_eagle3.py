"""Mobilint Llama EAGLE-3 model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintEagle3Cache
from ...utils.eagle3.eagle3_utils import (
    MobilintEagle3BaseModelMixin,
    MobilintEagle3DraftModelMixin,
    MobilintEagle3FCProjector,
    MobilintEagle3ModelMixin,
)
from ...utils.generation_utils import MobilintEagle3GenerationMixin, llm_eagle3_forward
from .configuration_llama_eagle3 import MobilintLlamaEagle3Config


class MobilintLlamaEagle3PreTrainedModel(PreTrainedModel):
    """Base pretrained model contract for Mobilint Llama EAGLE-3."""

    config: MobilintLlamaEagle3Config
    base_model_prefix = "model"
    main_input_name = "input_ids"


class MobilintLlamaEagle3BaseModel(MobilintEagle3BaseModelMixin, MobilintEagle3ModelMixin):
    """Concrete Llama base backend for EAGLE-3."""

    npu_backend_prefix = "base_"


class MobilintLlamaEagle3DraftModel(MobilintEagle3DraftModelMixin, MobilintEagle3ModelMixin):
    """Concrete Llama 1-block draft backend for Llama EAGLE-3."""

    npu_backend_prefix = "draft_"


class MobilintLlamaEagle3ForCausalLM(
    MobilintEagle3GenerationMixin,
    PretrainedOnlyMixin,
    MobilintLlamaEagle3PreTrainedModel,
):
    """Top-level Mobilint Llama EAGLE-3 causal LM.

    Generation compatibility notes:
    - Ignored with warning: ``attention_mask``, ``min_new_tokens``,
      ``pad_token_id``, ``npu_prefill_chunk_size``, ``cache_position``,
      and unknown ``generate`` kwargs.
    - Not supported (hard error): beam search, ``assistant_model``,
      ``use_cache=False``, custom ``logits_processor``, and negative prompts.
    - ``max_new_tokens`` resolution priority:
      1) explicit ``max_new_tokens`` argument,
      2) ``generation_config.max_new_tokens``,
      3) ``generation_config.max_length - prompt_length``,
      4) ``config.max_position_embeddings - prompt_length``.
    """

    config_class = MobilintLlamaEagle3Config

    def __init__(self, config: MobilintLlamaEagle3Config, *args: object, **kwargs: object) -> None:
        no_launch = bool(kwargs.pop("no_launch", False))
        super().__init__(config, *args, **kwargs)
        fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=no_launch)
        self.eagle3_fc_projector = fc_projector
        self.eagle3_base_model = MobilintLlamaEagle3BaseModel(config, _internal_call=True, no_launch=no_launch)
        self.eagle3_draft_model = MobilintLlamaEagle3DraftModel(
            config,
            draft_config=config.draft_config,
            fc_projector=fc_projector,
            _internal_call=True,
            no_launch=no_launch,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.eagle3_base_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: MobilintEagle3Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        count_npu_time: bool = False,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        """Run EAGLE-3 forward by delegating shared logic to utility helper."""
        return llm_eagle3_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            count_npu_time=count_npu_time,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )


AutoModel.register(MobilintLlamaEagle3Config, MobilintLlamaEagle3ForCausalLM)
AutoModelForCausalLM.register(MobilintLlamaEagle3Config, MobilintLlamaEagle3ForCausalLM)
