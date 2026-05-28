"""Utilities for Mobilint EAGLE-3 runtime and tree decoding."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_utils import PreTrainedModel

from .cache_utils import MobilintEagle3Cache
from .modeling_utils import MobilintEagle3ModelMixin

if TYPE_CHECKING:
    from ..models.qwen2_eagle3.configuration_qwen2_eagle3 import MobilintEagle3DraftConfig, MobilintQwen2Eagle3Config


class MobilintEagle3GenerationProtocol(Protocol):
    """Protocol for models that expose the EAGLE-3 generation contract."""

    eagle3_base_model: Any
    eagle3_draft_model: Any
    eagle3_fc_projector: Any


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
        self.position_table = None
        if self.inv_freq.device.type != "meta":
            self._build_position_table()

    def _build_position_table(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
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
        if self.position_table is None or seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_position_table(device=x.device, dtype=x.dtype)
        indices = position_ids.view(-1).cpu().numpy()
        assert self.position_table is not None
        return self.position_table[indices][None, None, :, :]


class MobilintEagle3FCProjector(MobilintEagle3ModelMixin, PreTrainedModel):
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


class MobilintEagle3BaseModelMixin:
    """Shared runtime helpers for an EAGLE-3 base model."""

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
            expanded_attn_mask = expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(inputs_embeds.device)
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
                logits_chunks.append(torch.from_numpy(logits).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).squeeze(1))
            elif seq_end == seq_length:
                logits_chunks.append(
                    torch.from_numpy(logits[:, :, -1]).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).squeeze(1)
                )

        hidden_states = torch.cat(hidden_states_chunks, dim=-2) if len(hidden_states_chunks) > 1 else hidden_states_chunks[0]
        logits_tensor = torch.cat(logits_chunks, dim=-2) if len(logits_chunks) > 1 else logits_chunks[0]
        return {"hidden_states": [hidden_states]}, logits_tensor if output_orig else hidden_states


class MobilintEagle3DraftModelMixin:
    """Shared runtime helpers for an EAGLE-3 draft model."""

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
            expanded_attn_mask = expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(hidden_states.device)
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
            attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device)
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
                .astype(np.float32, copy=False)
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
                hidden_chunks.append(torch.from_numpy(layer_outputs).to(device=hidden_states.device, dtype=hidden_states.dtype).squeeze(1))
                logits_chunks.append(
                    torch.from_numpy(last_hidden_logits).to(device=hidden_states.device, dtype=hidden_states.dtype).squeeze(1)
                )
            elif seq_end == seq_length:
                hidden_chunks.append(
                    torch.from_numpy(layer_outputs[:, :, -1]).to(device=hidden_states.device, dtype=hidden_states.dtype).squeeze(1)
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
        total_tokens = self.max_draft_tokens if max_draft_tokens is None else min(self.max_draft_tokens, max(1, max_draft_tokens))
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
        tree_mask = self.tree_mask_init.to(hidden_states.device)
        topk_cs_index = torch.arange(top_k, device=hidden_states.device)
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
            if self.draft_config.vocab_size == getattr(self.draft_config, "draft_vocab_size", self.draft_config.vocab_size):
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
        non_leaf_index = set(torch.unique(mask_index).tolist())
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


def load_embedding_override(path: str) -> torch.Tensor:
    """Load an embedding override from a torch tensor or state dict file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        if "weight" in data:
            return data["weight"]
        return next(iter(data.values()))
    raise ValueError(f"Unsupported embedding override payload in {path!r}")


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> Optional[LogitsProcessorList]:
    """Build a minimal logits processor list for EAGLE-3 generation."""
    processor_list = LogitsProcessorList()
    if temperature <= 1e-5:
        return None
    if temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list or None


def softmax_topk_cpu_torch(
    logits: torch.Tensor,
    k: int,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k probabilities and indices from logits."""
    x = logits.float()
    topk_vals, topk_idx = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    if logits_processor is not None:
        topk_vals = logits_processor(None, topk_vals)
    max_val = x.max(dim=-1, keepdim=True).values
    denom = torch.exp(x - max_val).sum(dim=-1, keepdim=True)
    probs = torch.exp(topk_vals - max_val) / denom
    return probs, topk_idx


@torch.no_grad()
def initialize_tree(
    input_ids: torch.LongTensor,
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    logits_processor: Optional[LogitsProcessorList],
    *,
    remaining_tokens: Optional[int] = None,
    count_npu_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill base model once and initialize the first draft tree."""
    outputs, logits = model.eagle3_base_model(
        input_ids[:, cache.get_base_seq_length() :],
        cache=cache,
        output_orig=True,
        requires_all_features_logits=False,
        count_npu_time=count_npu_time,
    )
    cache.update_base_seen_tokens(input_ids.shape[1] - cache.get_base_seq_length())

    if logits_processor is not None:
        probabilities, indices = softmax_topk_cpu_torch(logits, 10, logits_processor=logits_processor)
        sampled_idx = torch.multinomial(probabilities, 1)
        token = indices.gather(dim=1, index=sampled_idx)
    else:
        token = torch.argmax(logits, dim=-1, keepdim=True)

    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    hidden_states = outputs["hidden_states"][0].contiguous()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.eagle3_draft_model.topk_generate(
        hidden_states,
        input_ids=input_ids,
        cache=cache,
        logits_processor=logits_processor,
        max_draft_tokens=None if remaining_tokens is None else max(1, remaining_tokens - 1),
        count_npu_time=count_npu_time,
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits


@torch.no_grad()
def tree_decoding(
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    tree_candidates: torch.LongTensor,
    input_ids: torch.LongTensor,
    retrieve_indices: torch.LongTensor,
    tree_position_ids: torch.LongTensor,
    *,
    count_npu_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run base-model tree decoding for the current draft tree."""
    if cache.accept_tokens is not None:
        tree_position_ids = tree_position_ids + cache.accept_tokens.shape[1]
        accept_position_ids = torch.arange(cache.accept_tokens.shape[1], device=tree_position_ids.device)
        tree_position_ids = torch.cat((accept_position_ids, tree_position_ids), dim=0)
        tree_candidates = torch.cat((cache.accept_tokens, tree_candidates), dim=1)
        accepted_prefix_length = cache.accept_tokens.shape[1]
    else:
        accepted_prefix_length = 0

    position_ids = tree_position_ids + input_ids.shape[1] - accepted_prefix_length
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    outputs, tree_logits = model.eagle3_base_model(
        tree_candidates,
        cache=cache,
        output_orig=True,
        position_ids=position_ids,
        requires_all_features_logits=True,
        count_npu_time=count_npu_time,
    )

    if cache.accept_tokens is not None:
        cache.update_base_seen_tokens(accepted_prefix_length)
        cache.accept_tokens = None

    hidden_state = outputs["hidden_states"][0]
    del retrieve_indices
    tree_logits = tree_logits[:, accepted_prefix_length:]
    logits = tree_logits[0]
    return logits, hidden_state[:, accepted_prefix_length:]


@torch.no_grad()
def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    retrieve_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Choose the best accepted branch and the next sampling distribution."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    def select_token_logits(source: torch.Tensor, index: torch.Tensor | int) -> torch.Tensor:
        """Select a token distribution without collapsing it to a scalar."""
        selected = source[index]
        if selected.ndim == 0:
            return source
        if selected.ndim > 1:
            return selected[0]
        return selected

    if logits_processor is None:
        path_positions = retrieve_indices[:, :-1].to(logits.device)
        safe_positions = path_positions.clamp_min(0)
        path_logits = logits[safe_positions]
        greedy_tokens = torch.argmax(path_logits, dim=-1)
        candidate_targets = candidates[:, 1:].to(logits.device)
        valid_mask = (path_positions >= 0) & (candidate_targets >= 0)
        posterior_mask = ((candidate_targets == greedy_tokens) & valid_mask).int()
        accepted_lengths = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = accepted_lengths.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(accepted_lengths).to(torch.long)
        sample_index = torch.clamp(accept_length, max=path_logits.shape[1] - 1)
        sample_p = path_logits[best_candidate, sample_index]
        return best_candidate, accept_length, sample_p, None

    accept_length = 1
    accept_prefix = candidates[0][:1]
    best_candidate = 0
    retrieve_idx = retrieve_indices[0, :1]
    sampled_indices: Optional[torch.Tensor] = None
    adjusted = False

    for idx in range(1, candidates.shape[1]):
        if idx != accept_length:
            break
        matching = (candidates[:, :accept_length] == accept_prefix).all(dim=1)
        topk_probs, topk_indices = softmax_topk_cpu_torch(
            select_token_logits(logits, retrieve_idx),
            10,
            logits_processor=logits_processor,
        )
        sampled_indices = topk_indices
        seen_tokens: set[int] = set()
        for candidate_idx in range(candidates.shape[0]):
            if not matching[candidate_idx]:
                continue
            token = candidates[candidate_idx, idx].item()
            if token in seen_tokens or token == -1:
                continue
            seen_tokens.add(token)
            mask = topk_indices == token
            token_prob = topk_probs[mask.nonzero(as_tuple=True)[0].item()] if mask.any() else 0.0
            if torch.rand((), device=topk_probs.device) <= token_prob:
                accept_prefix = torch.cat((accept_prefix, candidates[candidate_idx, idx][None]), dim=0)
                accept_length += 1
                best_candidate = candidate_idx
                retrieve_idx = retrieve_indices[candidate_idx, idx]
                break
            if mask.any():
                topk_probs[mask.nonzero(as_tuple=True)[0].item()] = 0
                topk_probs = topk_probs / topk_probs.sum()
                adjusted = True

    if adjusted and accept_length != candidates.shape[1]:
        sample_p = topk_probs
    else:
        sample_logits = select_token_logits(logits, retrieve_indices[best_candidate, accept_length - 1])
        sample_p, sampled_indices = softmax_topk_cpu_torch(
            sample_logits,
            10,
            logits_processor=logits_processor,
        )

    return torch.tensor(best_candidate), torch.tensor(accept_length - 1), sample_p, sampled_indices


@torch.no_grad()
def update_inference_inputs(
    input_ids: torch.LongTensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accept_length: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    new_token_count: int,
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor,
    sampled_indices: Optional[torch.Tensor],
    *,
    remaining_tokens: Optional[int] = None,
    eos_token_id: Optional[int | list[int]] = None,
    count_npu_time: bool = False,
) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    """Advance accepted tokens and build the next draft tree."""
    accept_length_int = int(accept_length.item())
    best_candidate_int = int(best_candidate.item())
    accepted = candidates[None, best_candidate_int, : accept_length_int + 1].to(input_ids.device)
    if remaining_tokens is not None:
        accepted = accepted[:, : max(0, remaining_tokens)]
    should_stop = accepted.numel() == 0
    if eos_token_id is not None and accepted.numel() > 0:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        eos_positions = [index for index, token in enumerate(accepted[0].tolist()) if token in eos_ids]
        if eos_positions:
            accepted = accepted[:, : eos_positions[0] + 1]
            should_stop = True
    input_ids = torch.cat([input_ids, accepted], dim=-1)
    cache.accept_tokens = accepted
    new_token_count += int(accepted.shape[1])
    if should_stop or (remaining_tokens is not None and int(accepted.shape[1]) >= remaining_tokens):
        return (
            input_ids,
            torch.empty(0, dtype=torch.long, device=input_ids.device),
            retrieve_indices,
            torch.empty(0),
            torch.empty(0),
            new_token_count,
            True,
        )

    retrieved_hidden_state = hidden_state_new[:, retrieve_indices]
    accepted_hidden_state = retrieved_hidden_state[:, best_candidate_int, : int(accepted.shape[1])]

    if logits_processor is not None:
        assert sampled_indices is not None
        sampled_idx = torch.multinomial(sample_p, 1)
        token = sampled_indices[None, sampled_idx]
    else:
        token = torch.argmax(sample_p, dim=-1, keepdim=True)[None]

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.eagle3_draft_model.topk_generate(
        accepted_hidden_state,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        cache=cache,
        logits_processor=logits_processor,
        max_draft_tokens=None if remaining_tokens is None else max(1, remaining_tokens - int(accepted.shape[1]) - 1),
        count_npu_time=count_npu_time,
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_count, False
