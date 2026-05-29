"""Mobilint Qwen2 EAGLE-3 model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_utils import PreTrainedModel

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintEagle3Cache
from ...utils.eagle3.eagle3_utils import (
    CachedRotaryEmbedding,
    ScaledCachedRotaryEmbedding,
    MobilintEagle3BaseModelMixin,
    MobilintEagle3DraftModelMixin,
    MobilintEagle3FCProjector,
    MobilintEagle3ModelMixin,
)
from ...utils.generation_utils import MobilintEagle3GenerationMixin, llm_eagle3_forward
from .configuration_qwen2_eagle3 import MobilintQwen2Eagle3Config


class MobilintQwen2Eagle3PreTrainedModel(PreTrainedModel):
    """Base pretrained model contract for Mobilint EAGLE-3."""

    config: MobilintQwen2Eagle3Config
    base_model_prefix = "model"
    main_input_name = "input_ids"


class MobilintQwen2Eagle3BaseModel(MobilintEagle3BaseModelMixin, MobilintEagle3ModelMixin):
    """Concrete Qwen2 base backend for EAGLE-3."""

    npu_backend_prefix = "base_"

    def __init__(self, config: MobilintQwen2Eagle3Config, *args: object, **kwargs: object) -> None:
        super().__init__(config, *args, **kwargs)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = ScaledCachedRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            config=config,
        )


class MobilintQwen2Eagle3DraftModel(MobilintEagle3DraftModelMixin, MobilintEagle3ModelMixin):
    """Concrete Qwen2 draft backend for EAGLE-3 tree expansion."""

    npu_backend_prefix = "draft_"

    def __init__(
        self,
        config: MobilintQwen2Eagle3Config,
        draft_config: Qwen2Config,
        fc_projector: MobilintEagle3FCProjector,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.draft_config = draft_config
        self.fc_projector = fc_projector
        self.embed_tokens = nn.Embedding(draft_config.vocab_size, draft_config.hidden_size, draft_config.pad_token_id)
        head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
        self.rotary_emb = CachedRotaryEmbedding(head_dim, draft_config.max_position_embeddings)
        self.top_k = int(config.eagle3_tree_top_k)
        self.max_draft_tokens = int(getattr(config, "num_assistant_tokens", 63)) - 1
        self.depth = int(config.eagle3_tree_depth)
        self.hidden_size = draft_config.hidden_size
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        draft_vocab_size = int(getattr(draft_config, "draft_vocab_size", draft_config.vocab_size))
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long, device="cpu"))
        self.register_buffer("t2d", torch.zeros(draft_config.vocab_size, dtype=torch.bool, device="cpu"))
        self.tree_mask_init = torch.eye(self.top_k, dtype=torch.float32, device="cpu")[None, None]
        self.position_ids = torch.zeros(self.top_k, dtype=torch.long, device="cpu")
        for param in self.embed_tokens.parameters():
            param.requires_grad = False


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
        fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=no_launch)
        self.eagle3_fc_projector = fc_projector
        self.eagle3_base_model = MobilintQwen2Eagle3BaseModel(config, _internal_call=True, no_launch=no_launch)
        self.eagle3_draft_model = MobilintQwen2Eagle3DraftModel(
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


AutoModel.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
AutoModelForCausalLM.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
