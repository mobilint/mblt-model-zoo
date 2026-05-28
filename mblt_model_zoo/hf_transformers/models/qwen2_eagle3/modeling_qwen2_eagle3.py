"""Mobilint Qwen2 EAGLE-3 model implementation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintEagle3Cache
from ...utils.eagle3_utils import MobilintEagle3BaseModel, MobilintEagle3DraftModel, MobilintEagle3FCProjector
from ...utils.generation_utils import MobilintEagle3GenerationMixin
from .configuration_qwen2_eagle3 import MobilintQwen2Eagle3Config


class MobilintQwen2Eagle3PreTrainedModel(PreTrainedModel):
    """Base pretrained model contract for Mobilint EAGLE-3."""

    config: MobilintQwen2Eagle3Config
    base_model_prefix = "model"
    main_input_name = "input_ids"


class MobilintQwen2Eagle3Model(PretrainedOnlyMixin, MobilintQwen2Eagle3PreTrainedModel):
    """Nested EAGLE-3 model composition."""

    def __init__(self, config: MobilintQwen2Eagle3Config, *args: object, **kwargs: object) -> None:
        no_launch = bool(kwargs.pop("no_launch", False))
        super().__init__(config, *args, **kwargs)
        self.fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=no_launch)
        self.base_model = MobilintEagle3BaseModel(config, _internal_call=True, no_launch=no_launch)
        self.draft_model = MobilintEagle3DraftModel(
            config,
            draft_config=config.draft_config,
            fc_projector=self.fc_projector,
            _internal_call=True,
            no_launch=no_launch,
        )
        self.post_init()

    @property
    def embed_tokens(self) -> nn.Module:
        return self._modules["base_model"].embed_tokens

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,
        cache: MobilintEagle3Cache,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_orig: bool = True,
        requires_all_features_logits: bool = True,
        count_npu_time: bool = False,
    ) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
        return self.base_model(
            input_ids=input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_orig=output_orig,
            requires_all_features_logits=requires_all_features_logits,
            count_npu_time=count_npu_time,
        )


class MobilintQwen2Eagle3ForCausalLM(
    PretrainedOnlyMixin,
    MobilintQwen2Eagle3PreTrainedModel,
    MobilintEagle3GenerationMixin,
):
    """Top-level Mobilint EAGLE-3 causal LM."""

    config_class = MobilintQwen2Eagle3Config

    def __init__(self, config: MobilintQwen2Eagle3Config, *args: object, **kwargs: object) -> None:
        no_launch = bool(kwargs.pop("no_launch", False))
        super().__init__(config, *args, **kwargs)
        self.model = MobilintQwen2Eagle3Model(config, _internal_call=True, no_launch=no_launch)
        self.lm_head = nn.Identity()
        self.post_init()


AutoModel.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
AutoModelForCausalLM.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
