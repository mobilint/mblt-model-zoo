"""Mobilint Qwen2 EAGLE-3 model implementation."""

from __future__ import annotations

import math
import time
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintEagle3Cache
from ...utils.eagle3_utils import (
    evaluate_posterior,
    initialize_tree,
    load_embedding_override,
    prepare_logits_processor,
    tree_decoding,
    update_inference_inputs,
)
from ...utils.generation_utils import MobilintEagle3GenerationMixin, with_mobilint_generation_signature
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen2_eagle3 import MobilintEagle3DraftConfig, MobilintQwen2Eagle3Config


def make_causal_mask(
    input_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Build a causal mask compatible with MXQ inputs."""
    batch_size, target_length = input_shape
    mask = torch.full((target_length, target_length), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(target_length, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """Expand a 2D attention mask to decoder shape."""
    batch_size, src_len = mask.size()
    target_length = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_length, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def convert_attention_mask_to_numpy(
    attention_mask: torch.Tensor,
    *,
    squeeze_channel_dim: bool = False,
) -> np.ndarray:
    """Convert a decoder attention mask to a float32 numpy payload."""
    if squeeze_channel_dim:
        attention_mask = attention_mask.squeeze(1)
    attention_mask = (attention_mask != 0).to(torch.float32)
    return attention_mask.contiguous().numpy()


class CachedRotaryEmbedding(nn.Module):
    """Precompute RoPE tables and expose them as MXQ-friendly numpy arrays."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_position_table()

    def _build_position_table(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        device = self.inv_freq.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        with torch.no_grad():
            seq_len = self.max_seq_len
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)
            channels = cos.shape[-1]
            half_channels = channels // 2
            rotate_tensor = torch.zeros(1, 1, seq_len, 2 * channels, device=device, dtype=dtype)
            rotate_tensor[..., 0:channels:2] = cos[..., :half_channels]
            rotate_tensor[..., 1:channels:2] = -sin[..., :half_channels]
            rotate_tensor[..., channels : 2 * channels : 2] = sin[..., half_channels:channels]
            rotate_tensor[..., channels + 1 : 2 * channels : 2] = cos[..., half_channels:channels]
            target_half = ((channels + 63) // 64) * 64
            target_size = 2 * target_half
            if rotate_tensor.shape[-1] != target_size:
                rotate_tensor = self.pad_rope(rotate_tensor, target_size)
            self.position_table = rotate_tensor.cpu().numpy()[0, 0]

    @staticmethod
    def pad_rope(rotate_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        batch_size, num_heads, seq_len, current_len = rotate_tensor.shape
        if target_len == current_len:
            return rotate_tensor
        if target_len < current_len:
            return rotate_tensor[..., :target_len]
        channels = current_len // 2
        target_half = target_len // 2
        pad_half = target_half - channels
        if pad_half <= 0:
            return rotate_tensor
        tensor = rotate_tensor.reshape(batch_size, num_heads, seq_len, 2, -1)
        padding = rotate_tensor.new_zeros(batch_size, num_heads, seq_len, 2, pad_half)
        tensor = torch.cat([tensor, padding], dim=4)
        return tensor.reshape(batch_size, num_heads, seq_len, target_len)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> np.ndarray:
        seq_len = int(torch.max(position_ids).item()) + 1
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_position_table(device=x.device, dtype=x.dtype)
        indices = position_ids.view(-1).cpu().numpy()
        return self.position_table[indices][None, None, :, :]


class MobilintEagle3ModelMixin(MobilintModelMixin):
    """Base Mobilint model mixin for EAGLE-3 child backends."""

    pass


class MobilintQwen2Eagle3PreTrainedModel(PreTrainedModel):
    """Base pretrained model contract for Mobilint EAGLE-3."""

    config: MobilintQwen2Eagle3Config
    base_model_prefix = "model"
    main_input_name = "input_ids"


class MobilintEagle3FCProjector(MobilintEagle3ModelMixin, MobilintQwen2Eagle3PreTrainedModel):
    """FC projection backend used by the draft model."""

    npu_backend_prefix = "fc_"

    def project(self, hidden_states: torch.Tensor, *, count_npu_time: bool = False) -> torch.Tensor:
        """Project base hidden states to the draft hidden size."""
        hidden_states_numpy = hidden_states.cpu().contiguous().numpy().astype(np.float32, copy=False)
        if hidden_states_numpy.ndim == 3:
            hidden_states_numpy = np.expand_dims(hidden_states_numpy, 1)
        if count_npu_time:
            start_time = time.perf_counter()
            result = self.get_mxq_model().infer([hidden_states_numpy])
            self._record_npu_timing("decode", time.perf_counter() - start_time)
        else:
            result = self.get_mxq_model().infer([hidden_states_numpy])
        assert result is not None, "mxq infer result is None!"
        return torch.from_numpy(result[0]).to(device=hidden_states.device, dtype=hidden_states.dtype).squeeze(1)


class MobilintEagle3BaseModel(MobilintEagle3ModelMixin, MobilintQwen2Eagle3PreTrainedModel):
    """Base Qwen2 MXQ wrapper for EAGLE-3."""

    npu_backend_prefix = "base_"

    def __init__(self, config: MobilintQwen2Eagle3Config, *args: object, **kwargs: object) -> None:
        super().__init__(config, *args, **kwargs)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = CachedRotaryEmbedding(head_dim, config.max_position_embeddings)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        cache: MobilintEagle3Cache,
    ) -> torch.Tensor:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                torch.Size(input_shape),
                torch.float32,
                inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        if cache.tree_mask is not None and combined_attention_mask is not None:
            tree_mask = cache.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = combined_attention_mask.min()
        assert combined_attention_mask is not None
        return combined_attention_mask

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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds.")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            assert inputs_embeds is not None
            batch_size, seq_length, _ = inputs_embeds.shape

        past_key_values_length = cache.get_base_seq_length()
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds_numpy = inputs_embeds.cpu().contiguous().float().numpy()
        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)

        position_embs_numpy = self.rotary_emb(inputs_embeds, position_ids).astype(np.float32, copy=False)
        if position_embs_numpy.ndim == 3:
            position_embs_numpy = np.expand_dims(position_embs_numpy, 1)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            cache,
        )
        attention_mask_numpy = convert_attention_mask_to_numpy(attention_mask, squeeze_channel_dim=True)
        if attention_mask_numpy.ndim == 3:
            attention_mask_numpy = np.expand_dims(attention_mask_numpy, 1)

        hidden_states_chunks: list[torch.Tensor] = []
        logits_chunks: list[torch.Tensor] = []
        chunk_size = self.resolve_prefill_chunk_size(self.config.eagle3_npu_chunk_size)
        timing_phase = "prefill" if seq_length > 1 or past_key_values_length == 0 else "decode"
        for chunk_index in range(math.ceil(seq_length / chunk_size)):
            seq_start = chunk_index * chunk_size
            seq_end = min((chunk_index + 1) * chunk_size, seq_length)
            current_cache_position = past_key_values_length + seq_start
            infer_inputs = [
                inputs_embeds_numpy[:, :, seq_start:seq_end, :],
                attention_mask_numpy[:, :, seq_start:seq_end, : current_cache_position + seq_end - seq_start],
                position_embs_numpy[:, :, seq_start:seq_end, :],
            ]
            if count_npu_time:
                start_time = time.perf_counter()
                result = self.get_mxq_model().infer(infer_inputs, None, current_cache_position)
                self._record_npu_timing(timing_phase, time.perf_counter() - start_time)
            else:
                result = self.get_mxq_model().infer(infer_inputs, None, current_cache_position)
            assert result is not None, "mxq infer result is None!"
            hidden1, hidden2, hidden3, logits = result
            hidden = torch.from_numpy(np.concatenate([hidden1, hidden2, hidden3], axis=-1)).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            if hidden.ndim == 4 and hidden.shape[1] == 1:
                hidden = hidden.squeeze(1)
            hidden_states_chunks.append(hidden)
            if requires_all_features_logits:
                logits_chunks.append(
                    torch.from_numpy(logits).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).squeeze(1)
                )
            elif seq_end == seq_length:
                logits_chunks.append(
                    torch.from_numpy(logits[:, :, -1])
                    .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    .squeeze(1)
                )

        hidden_states = (
            torch.cat(hidden_states_chunks, dim=-2) if len(hidden_states_chunks) > 1 else hidden_states_chunks[0]
        )
        logits_tensor = torch.cat(logits_chunks, dim=-2) if len(logits_chunks) > 1 else logits_chunks[0]
        return {"hidden_states": [hidden_states]}, logits_tensor if output_orig else hidden_states


class MobilintEagle3DraftModel(MobilintEagle3ModelMixin, MobilintQwen2Eagle3PreTrainedModel):
    """Draft Qwen2 wrapper for EAGLE-3 tree expansion."""

    npu_backend_prefix = "draft_"

    def __init__(
        self,
        config: MobilintQwen2Eagle3Config,
        draft_config: MobilintEagle3DraftConfig,
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
        self.total_tokens = int(getattr(config, "num_assistant_tokens", 63)) - 1
        self.depth = int(config.eagle3_tree_depth)
        self.hidden_size = draft_config.hidden_size
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        draft_vocab_size = int(getattr(draft_config, "draft_vocab_size", draft_config.vocab_size))
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(draft_config.vocab_size, dtype=torch.bool))
        self.tree_mask_init = torch.eye(self.top_k, dtype=torch.float32)[None, None]
        self.position_ids = torch.zeros(self.top_k, dtype=torch.long)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        hidden_states: torch.Tensor,
        past_key_values_length: int,
        cache: MobilintEagle3Cache,
        tree_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                torch.Size(input_shape),
                torch.float32,
                hidden_states.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                hidden_states.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        if tree_mask is not None and combined_attention_mask is not None:
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][tree_mask == 0] = torch.finfo(torch.float32).min
        assert combined_attention_mask is not None
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.LongTensor,
        cache: MobilintEagle3Cache,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        requires_all_features: bool = False,
        add_cache_position: int = 0,
        tree_mask: Optional[torch.Tensor] = None,
        count_npu_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = cache.get_draft_seq_length() + add_cache_position
        seq_length_with_past = seq_length + past_key_values_length

        inputs_embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)
        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        position_embs_numpy = self.rotary_emb(inputs_embeds, position_ids).astype(np.float32, copy=False)
        if position_embs_numpy.ndim == 3:
            position_embs_numpy = np.expand_dims(position_embs_numpy, 1)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
            cache,
            tree_mask,
        )
        attention_mask_numpy = convert_attention_mask_to_numpy(attention_mask)
        if attention_mask_numpy.ndim == 3:
            attention_mask_numpy = np.expand_dims(attention_mask_numpy, 1)

        inputs_embeds_numpy = inputs_embeds.cpu().contiguous().float().numpy()
        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)
        hidden_states_numpy = hidden_states.cpu().contiguous().numpy().astype(np.float32, copy=False)
        if hidden_states_numpy.ndim == 3:
            hidden_states_numpy = np.expand_dims(hidden_states_numpy, 1)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states_numpy = (
                self.fc_projector.project(hidden_states, count_npu_time=count_npu_time)
                .cpu()
                .contiguous()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )
            if hidden_states_numpy.ndim == 3:
                hidden_states_numpy = np.expand_dims(hidden_states_numpy, 1)

        chunk_size = self.resolve_prefill_chunk_size(self.config.eagle3_npu_chunk_size)
        hidden_chunks: list[torch.Tensor] = []
        logits_chunks: list[torch.Tensor] = []
        base_cache_position = cache.get_draft_seq_length() + add_cache_position
        timing_phase = "prefill" if seq_length > 1 or base_cache_position == 0 else "decode"
        for chunk_index in range(math.ceil(seq_length / chunk_size)):
            seq_start = chunk_index * chunk_size
            seq_end = min((chunk_index + 1) * chunk_size, seq_length)
            current_cache_position = base_cache_position + seq_start
            infer_inputs = [
                hidden_states_numpy[:, :, seq_start:seq_end, :],
                inputs_embeds_numpy[:, :, seq_start:seq_end, :],
                attention_mask_numpy[:, :, seq_start:seq_end, : current_cache_position + seq_end - seq_start],
                position_embs_numpy[:, :, seq_start:seq_end, :],
            ]
            if count_npu_time:
                start_time = time.perf_counter()
                result = self.get_mxq_model().infer(infer_inputs, None, current_cache_position)
                self._record_npu_timing(timing_phase, time.perf_counter() - start_time)
            else:
                result = self.get_mxq_model().infer(infer_inputs, None, current_cache_position)
            assert result is not None, "mxq infer result is None!"
            layer_outputs, last_hidden_logits = result
            if requires_all_features:
                hidden_chunks.append(
                    torch.from_numpy(layer_outputs)
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .squeeze(1)
                )
                logits_chunks.append(
                    torch.from_numpy(last_hidden_logits)
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .squeeze(1)
                )
            elif seq_end == seq_length:
                hidden_chunks.append(
                    torch.from_numpy(layer_outputs[:, :, -1])
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .squeeze(1)
                )
                logits_chunks.append(
                    torch.from_numpy(last_hidden_logits[:, :, -1])
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .squeeze(1)
                )

        hidden_out = torch.cat(hidden_chunks, dim=-2)
        logits_out = torch.cat(logits_chunks, dim=-1)
        return hidden_out, logits_out

    @torch.no_grad()
    def topk_generate(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.LongTensor,
        cache: MobilintEagle3Cache,
        logits_processor: Optional[Any],
        max_draft_tokens: Optional[int] = None,
        count_npu_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens = self.total_tokens if max_draft_tokens is None else min(self.total_tokens, max(1, max_draft_tokens))
        depth = self.depth
        top_k = self.top_k
        sample_token = input_ids[:, -1]
        scores_list = []
        parents_list = []
        draft_token_steps = []

        input_ids_without_prompt_token = input_ids[:, 1:].to(hidden_states.device)
        input_ids_delta = input_ids_without_prompt_token[:, cache.get_draft_seq_length() :]
        last_hidden, last_hidden_logits = self(
            hidden_states,
            input_ids=input_ids_delta,
            cache=cache,
            requires_all_features=False,
            add_cache_position=0,
            tree_mask=None,
            count_npu_time=count_npu_time,
        )
        cache.update_draft_seen_tokens(input_ids_delta.shape[1])

        last_p = self.logsoftmax(last_hidden_logits)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.draft_config.vocab_size == getattr(self.draft_config, "draft_vocab_size", self.draft_config.vocab_size):
            draft_token_steps.append(topk_index)
            next_input_ids = topk_index
        else:
            draft_token_steps.append(topk_index + self.d2t[topk_index])
            next_input_ids = topk_index + self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        add_cache_position = 0
        length_position = input_ids.shape[1] - 1

        for depth_index in range(depth):
            position_ids = length_position + self.position_ids.to(hidden_states.device)
            out_hidden, last_hidden_logits = self(
                input_hidden,
                input_ids=next_input_ids,
                cache=cache,
                position_ids=position_ids,
                requires_all_features=True,
                add_cache_position=add_cache_position,
                tree_mask=tree_mask,
                count_npu_time=count_npu_time,
            )
            length_position += 1
            bias1 = top_k if depth_index > 0 else 0
            bias2 = max(0, depth_index - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)
            last_hidden_logits = last_hidden_logits.squeeze(0)
            last_p = self.logsoftmax(last_hidden_logits)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            cumulative_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cumulative_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            next_input_ids = topk_index.view(-1)[topk_cs_index][None]
            if self.draft_config.vocab_size == getattr(
                self.draft_config, "draft_vocab_size", self.draft_config.vocab_size
            ):
                draft_token_steps.append(topk_index)
            else:
                next_input_ids = next_input_ids + self.d2t[next_input_ids]
                draft_token_steps.append(topk_index + self.d2t[topk_index])
            scores_list.append(cumulative_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init.to(tree_mask.device)), dim=3)
            add_cache_position += self.top_k

        scores_list_tensor = torch.cat(scores_list, dim=0).view(-1)
        draft_step_tensor = torch.cat(draft_token_steps, dim=0).view(-1)
        top_scores = torch.topk(scores_list_tensor, total_tokens, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
        draft_tokens = draft_step_tensor[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask_tensor = torch.eye(total_tokens + 1).bool()
        tree_mask_tensor[:, 0] = True
        for token_index in range(total_tokens):
            tree_mask_tensor[token_index + 1].add_(tree_mask_tensor[mask_index_list[token_index]])
        tree_position_ids = torch.sum(tree_mask_tensor, dim=1) - 1
        tree_mask_tensor = tree_mask_tensor.float()[None, None]
        draft_tokens = draft_tokens[None]

        max_depth = torch.max(tree_position_ids) + 1
        non_leaf_index = torch.unique(mask_index).tolist()
        leaf_count = total_tokens - (len(non_leaf_index) - 1)
        retrieve_indices = torch.zeros(leaf_count, max_depth.item(), dtype=torch.long) - 1
        retrieve_index_rows = retrieve_indices.tolist()
        row_id = 0
        position_ids_list = tree_position_ids.tolist()
        for token_index in range(total_tokens + 1):
            if token_index not in non_leaf_index:
                current_id = token_index
                node_depth = position_ids_list[token_index]
                for depth_index in reversed(range(node_depth + 1)):
                    retrieve_index_rows[row_id][depth_index] = current_id
                    current_id = mask_index_list[current_id - 1]
                row_id += 1
        if logits_processor is not None:
            max_item = total_tokens + 5

            def _sort_key(values: list[int]) -> list[int]:
                return [value if value >= 0 else max_item for value in values]

            retrieve_index_rows = sorted(retrieve_index_rows, key=_sort_key)
        retrieve_indices = torch.tensor(retrieve_index_rows, dtype=torch.long)
        return (
            draft_tokens.to(hidden_states.device),
            retrieve_indices.to(hidden_states.device),
            tree_mask_tensor.to(hidden_states.device),
            tree_position_ids.to(hidden_states.device),
        )


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

    @property
    def eagle3_model(self) -> MobilintQwen2Eagle3Model:
        return self._modules["model"]

    @property
    def eagle3_base_model(self) -> MobilintEagle3BaseModel:
        return self.eagle3_model._modules["base_model"]

    @property
    def eagle3_draft_model(self) -> MobilintEagle3DraftModel:
        return self.eagle3_model._modules["draft_model"]

    @staticmethod
    def _clear_tree_state(cache: Any) -> None:
        """Clear transient EAGLE-3 tree state on any cache-like object."""
        clear_tree_state = getattr(cache, "clear_tree_state", None)
        if callable(clear_tree_state):
            clear_tree_state()
            return
        for attribute in (
            "accept_tokens",
            "tree_mask",
            "retrieve_indices",
            "tree_position_ids",
            "pending_draft_tokens",
        ):
            if hasattr(cache, attribute):
                setattr(cache, attribute, None)

    @staticmethod
    def _sync_draft_seq_length_to_base(cache: Any) -> None:
        """Align any draft cache length metadata with the committed base length."""
        get_base_seq_length = getattr(cache, "get_base_seq_length", None)
        set_draft_seq_length = getattr(cache, "set_draft_seq_length", None)
        if callable(get_base_seq_length) and callable(set_draft_seq_length):
            try:
                set_draft_seq_length(get_base_seq_length())
            except AttributeError:
                return
            return
        draft_layer = getattr(cache, "draft_layer", None)
        if draft_layer is not None and callable(get_base_seq_length) and hasattr(draft_layer, "set_seq_length"):
            try:
                draft_layer.set_seq_length(get_base_seq_length())
            except AttributeError:
                return

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str], *model_args: object, **kwargs: object):
        legacy_embedding_weight = kwargs.pop("embedding_weight", None)
        base_embedding_weight = kwargs.pop("base_embedding_weight", None)
        draft_embedding_weight = kwargs.pop("draft_embedding_weight", None)
        if legacy_embedding_weight is not None:
            raise ValueError(
                "`embedding_weight` is not supported for mobilint-qwen2-eagle3. "
                "Use `base_embedding_weight` and/or `draft_embedding_weight`."
            )
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if base_embedding_weight is not None:
            model._inject_embedding_override(model.eagle3_base_model.embed_tokens, base_embedding_weight, "base")
        if draft_embedding_weight is not None:
            model._inject_embedding_override(model.eagle3_draft_model.embed_tokens, draft_embedding_weight, "draft")
        return model

    @staticmethod
    def _inject_embedding_override(embedding: nn.Embedding, path: str, role: str) -> None:
        weight = load_embedding_override(path)
        if weight.ndim != 2:
            raise ValueError(f"{role} embedding override must be rank 2, got shape {tuple(weight.shape)}")
        expected_shape = tuple(embedding.weight.shape)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"{role} embedding override shape mismatch: expected {expected_shape}, got {tuple(weight.shape)}"
            )
        with torch.no_grad():
            embedding.weight.data = weight.to(device=embedding.weight.device, dtype=embedding.weight.dtype)

    def get_input_embeddings(self) -> nn.Module:
        return self.eagle3_model.get_input_embeddings()

    def get_cache_mxq_models(self) -> tuple[Any, Any]:
        return self.eagle3_base_model.get_mxq_model(), self.eagle3_draft_model.get_mxq_model()

    def reset_npu_timing(self) -> None:
        """Reset aggregate NPU timing counters for all EAGLE-3 child backends."""
        for child in (self.eagle3_base_model, self.eagle3_draft_model, self.eagle3_model.fc_projector):
            child.reset_npu_timing()

    def get_npu_timing(self) -> dict[str, float | int]:
        """Return aggregate NPU timing counters across base, draft, and FC backends."""
        aggregate: dict[str, float | int] = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_calls": 0,
            "decode_calls": 0,
        }
        for child in (self.eagle3_base_model, self.eagle3_draft_model, self.eagle3_model.fc_projector):
            timing = child.get_npu_timing()
            aggregate["prefill_time"] = float(aggregate["prefill_time"]) + float(timing.get("prefill_time", 0.0))
            aggregate["decode_time"] = float(aggregate["decode_time"]) + float(timing.get("decode_time", 0.0))
            aggregate["prefill_calls"] = int(aggregate["prefill_calls"]) + int(timing.get("prefill_calls", 0))
            aggregate["decode_calls"] = int(aggregate["decode_calls"]) + int(timing.get("decode_calls", 0))
        return aggregate

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintEagle3Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        count_npu_time: bool = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del cache_position, output_hidden_states, output_attentions, kwargs
        if use_cache is False:
            raise ValueError("mobilint-qwen2-eagle3 requires use_cache=True.")
        if past_key_values is None:
            past_key_values = self._get_cache("mobilint-eagle3", 1, 0)
        if not isinstance(past_key_values, MobilintEagle3Cache):
            raise TypeError("past_key_values must be MobilintEagle3Cache for mobilint-qwen2-eagle3.")
        outputs, logits = self.eagle3_model(
            input_ids=input_ids,
            cache=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_orig=True,
            requires_all_features_logits=False,
            count_npu_time=count_npu_time,
        )
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None if outputs is None else tuple(outputs["hidden_states"]),
            attentions=None,
        )

    @torch.no_grad()
    @with_mobilint_generation_signature(
        GenerationMixin.generate,
        "count_npu_time",
        "prefill_chunk_size",
    )
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintEagle3Cache] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        streamer: Optional[Any] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        num_beams: int = 1,
        assistant_model: Optional[PreTrainedModel] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        stopping_criteria: Optional[StoppingCriteriaList | list[Any]] = None,
        min_new_tokens: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int | list[int]] = None,
        count_npu_time: bool = False,
        prefill_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor | GenerateDecoderOnlyOutput:
        """Generate tokens with the Mobilint EAGLE-3 decoding loop.

        Args:
            inputs: Optional tensor alias for ``input_ids``.
            input_ids: Prompt token ids.
            attention_mask: Optional attention mask for the prompt.
            past_key_values: Optional Mobilint EAGLE-3 cache for continuation.
            generation_config: Optional Hugging Face generation config from the pipeline.
            max_new_tokens: Maximum number of new tokens to emit.
            do_sample: Whether to use sampling. ``False`` forces deterministic decoding.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            streamer: Optional text streamer passed through by the pipeline.
            return_dict_in_generate: Whether to return a Hugging Face generation output object.
            output_scores: Unsupported Hugging Face generation output mode.
            output_hidden_states: Unsupported Hugging Face generation output mode.
            output_attentions: Unsupported Hugging Face generation output mode.
            num_beams: Beam search width. Only greedy decoding is supported.
            assistant_model: Unsupported Hugging Face assistant model.
            use_cache: Compatibility flag forwarded by Hugging Face generation helpers.
            cache_position: Compatibility cache position forwarded by Hugging Face generation helpers.
            **kwargs: Additional generation kwargs.

        Returns:
            The generated token tensor, or a Hugging Face generation output when requested.
        """
        if output_scores or output_hidden_states or output_attentions:
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support generation diagnostics yet.")
        if num_beams != 1:
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support beam search.")
        if assistant_model is not None:
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support HF assistant_model mixing.")
        if use_cache is False:
            raise NotImplementedError("mobilint-qwen2-eagle3 requires use_cache=True.")
        del attention_mask, min_new_tokens, pad_token_id, prefill_chunk_size
        logits_processor_arg = kwargs.pop("logits_processor", None)
        synced_gpus = kwargs.pop("synced_gpus", None)
        negative_prompt_ids = kwargs.pop("negative_prompt_ids", None)
        negative_prompt_attention_mask = kwargs.pop("negative_prompt_attention_mask", None)
        if synced_gpus not in (None, False):
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support synced_gpus generation.")
        if logits_processor_arg not in (None, []):
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support custom logits_processor yet.")
        if negative_prompt_ids is not None or negative_prompt_attention_mask is not None:
            raise NotImplementedError("mobilint-qwen2-eagle3 does not support negative prompts.")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise NotImplementedError(f"Unsupported generate kwargs for mobilint-qwen2-eagle3: {unsupported}")

        input_ids = input_ids if input_ids is not None else inputs
        if input_ids is None:
            raise ValueError("`generate` requires `input_ids` or `inputs`.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("mobilint-qwen2-eagle3 only supports batch size 1.")
        if cache_position is not None:
            del cache_position

        generation_config = self.generation_config if generation_config is None else generation_config
        resolved_max_new_tokens = int(
            max_new_tokens if max_new_tokens is not None else generation_config.max_new_tokens
        )
        resolved_do_sample = bool(do_sample if do_sample is not None else getattr(generation_config, "do_sample", False))
        resolved_temperature = temperature if temperature is not None else getattr(generation_config, "temperature", None)
        resolved_top_p = top_p if top_p is not None else getattr(generation_config, "top_p", None)
        resolved_top_k = int(top_k if top_k is not None else getattr(generation_config, "top_k", 0))
        if not resolved_do_sample:
            resolved_temperature = 0.0
        num_assistant_tokens = int(getattr(generation_config, "num_assistant_tokens", 64))
        self.eagle3_draft_model.total_tokens = max(1, num_assistant_tokens - 1)

        cache = past_key_values
        if cache is None:
            cache = self._get_cache("mobilint-eagle3", 1, 0)
            cache.reset()
        elif not isinstance(cache, MobilintEagle3Cache):
            raise TypeError("past_key_values must be MobilintEagle3Cache for mobilint-qwen2-eagle3.")
        self._clear_tree_state(cache)
        self._sync_draft_seq_length_to_base(cache)
        logits_processor = prepare_logits_processor(
            temperature=0.0 if resolved_temperature is None else resolved_temperature,
            top_p=0.0 if resolved_top_p is None else resolved_top_p,
            top_k=resolved_top_k,
        )

        generated = input_ids.clone()
        eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
        if stopping_criteria is None:
            stopping_criteria_list = StoppingCriteriaList()
        elif isinstance(stopping_criteria, StoppingCriteriaList):
            stopping_criteria_list = stopping_criteria
        else:
            stopping_criteria_list = StoppingCriteriaList(stopping_criteria)
        if streamer is not None:
            streamer.put(generated[0].detach().cpu())
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = initialize_tree(
            generated,
            self,
            cache,
            logits_processor,
            remaining_tokens=resolved_max_new_tokens,
            count_npu_time=count_npu_time,
        )
        new_token_count = 0

        while new_token_count < resolved_max_new_tokens:
            remaining_tokens = resolved_max_new_tokens - new_token_count
            logits, hidden_state_new = tree_decoding(
                self,
                cache,
                draft_tokens.to(generated.device),
                generated,
                retrieve_indices,
                tree_position_ids,
                count_npu_time=count_npu_time,
            )
            padding = torch.full((1, 1), -1, dtype=torch.long, device=generated.device)
            padded_draft_tokens = torch.cat((draft_tokens.to(generated.device), padding), dim=1)
            candidates = padded_draft_tokens[0, retrieve_indices].contiguous()
            best_candidate, accept_length, sample_p, sampled_indices = evaluate_posterior(
                logits,
                candidates,
                logits_processor,
                retrieve_indices,
            )
            prev_len = generated.shape[1]
            generated, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_count, should_stop = (
                update_inference_inputs(
                    generated,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token_count,
                    self,
                    cache,
                    hidden_state_new,
                    sample_p,
                    sampled_indices,
                    remaining_tokens=remaining_tokens,
                    eos_token_id=eos_token_id,
                    count_npu_time=count_npu_time,
                )
            )
            if streamer is not None:
                for token_id in generated[0, prev_len:]:
                    streamer.put(token_id.unsqueeze(0))
            if stopping_criteria_list(generated, sample_p):
                break
            if should_stop:
                break

        if streamer is not None:
            streamer.end()
        self._clear_tree_state(cache)
        self._sync_draft_seq_length_to_base(cache)
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=generated, past_key_values=cache)
        return generated


AutoModel.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
AutoModelForCausalLM.register(MobilintQwen2Eagle3Config, MobilintQwen2Eagle3ForCausalLM)
