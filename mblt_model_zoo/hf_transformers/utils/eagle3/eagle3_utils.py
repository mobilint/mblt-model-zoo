"""Utilities for Mobilint EAGLE-3 runtime and tree decoding.

This module contains the low-level building blocks for the EAGLE-3 decoding pipeline:

1) Base prefill/decode (`MobilintEagle3BaseModelMixin`)
2) Draft tree expansion (`MobilintEagle3DraftModelMixin.topk_generate`)
3) Posterior evaluation and accepted-token update

Typical tensor shapes (batch is currently fixed to 1):

- ``input_ids``: ``[1, prompt_len]``
- Base logits (single-step): ``[1, vocab]``
- Draft tokens: ``[1, tree_nodes]`` (first token is the sampled root)
- ``retrieve_indices``: ``[leaf_count, depth]``
- Tree mask: ``[1, 1, tree_nodes, tree_nodes]``
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from ..cache_utils import MobilintEagle3Cache
from ..modeling_utils import MobilintEagle3ModelMixin

logger = logging.get_logger(__name__)


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
    """Build a causal mask compatible with MXQ inputs.

    Shape examples:
    - input_shape ``(1, 4)``, ``past_key_values_length=8``
      -> output ``[1, 1, 4, 12]``
    """
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
    """Convert a decoder attention mask to a float32 numpy payload.

    The MXQ runtime consumes dense float arrays where non-zero means masked.
    We keep the same logical layout and convert to contiguous ``np.float32``.
    """
    if squeeze_channel_dim:
        attention_mask = attention_mask.squeeze(1)
    attention_mask = (attention_mask != 0).to(torch.float32)
    return attention_mask.cpu().contiguous().numpy()


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


class ScaledCachedRotaryEmbedding(nn.Module):
    """RoPE table cache supporting config-driven scaling for base MXQ models.

    This implementation mirrors JPharmatron-style base RoPE behavior:
    - supports ``config.rope_scaling`` via ``ROPE_INIT_FUNCTIONS``
    - applies runtime ``attention_scaling`` to cos/sin tables
    - keeps MXQ-friendly output shape ``[B, 1, T, pe_size]``
    """

    def __init__(
        self,
        dim: int | None = None,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        rope_type: str = "default",
        config: Optional[Any] = None,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.rope_kwargs: dict[str, Any] = {}
        self.dim = dim
        self.base = base

        if config is None:
            self.rope_type = rope_type
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": 1.0,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.max_seq_len = (
                max_position_embeddings if max_length is None or max_length < max_position_embeddings else max_length
            )
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is not None:
                self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
                self.rope_kwargs = dict(rope_scaling)
            else:
                self.rope_type = "default"
            config_max_pos = int(getattr(config, "max_position_embeddings", max_position_embeddings))
            self.max_seq_len = config_max_pos if max_length is None or max_length < config_max_pos else max_length

        if "rope_type" not in self.rope_kwargs:
            self.rope_kwargs["rope_type"] = self.rope_type
        if dim is not None and "dim" not in self.rope_kwargs:
            self.rope_kwargs["dim"] = dim
        if "base" not in self.rope_kwargs:
            self.rope_kwargs["base"] = base
        if "max_position_embeddings" not in self.rope_kwargs:
            self.rope_kwargs["max_position_embeddings"] = self.max_seq_len

        self.original_max_seq_len = self.max_seq_len
        self.rope_init_fn = self._resolve_rope_init_fn()
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, None, seq_len=self.max_seq_len, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq.clone()
        self.position_table: Optional[np.ndarray] = None
        if self.inv_freq.device.type != "meta":
            self._build_position_table()

    def _resolve_rope_init_fn(self):
        if self.rope_type in (None, "default"):
            return self.compute_default_rope_parameters
        return ROPE_INIT_FUNCTIONS[self.rope_type]

    def compute_default_rope_parameters(
        self,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        return self._compute_default_rope_parameters(config=config or self.config, device=device, **kwargs)

    def _compute_default_rope_parameters(
        self,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, float]:
        del kwargs
        resolved_config = config or self.config
        head_dim = self.dim
        if head_dim is None and resolved_config is not None:
            head_dim = getattr(resolved_config, "head_dim", resolved_config.hidden_size // resolved_config.num_attention_heads)
        if head_dim is None:
            raise ValueError("ScaledCachedRotaryEmbedding requires `dim` or a config with head-dimension metadata.")

        partial_rotary_factor = 1.0
        if resolved_config is not None:
            partial_rotary_factor = float(getattr(resolved_config, "partial_rotary_factor", 1.0))
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope_theta = self.base
        if resolved_config is not None:
            rope_theta = float(getattr(resolved_config, "rope_theta", rope_theta))

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / float(rotary_dim))
        )
        return inv_freq, 1.0

    def _build_position_table(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        device = self.inv_freq.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        with torch.no_grad():
            seq_len = self.max_seq_len
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            inv_freq_expanded = self.inv_freq[None, :, None].float()
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
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
                rotate_tensor = CachedRotaryEmbedding.pad_rope(rotate_tensor, target_size)
            self.position_table = rotate_tensor.cpu().numpy()[0, 0]

    def _dynamic_frequency_update(self, position_ids: torch.LongTensor, device: torch.device) -> None:
        if self.rope_type in (None, "default"):
            return
        seq_len = int(torch.max(position_ids).item()) + 1
        if seq_len > self.max_seq_len:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len = seq_len
            self._build_position_table(device=self.inv_freq.device)
        if seq_len < self.original_max_seq_len and self.max_seq_len > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len = self.original_max_seq_len
            self._build_position_table(device=self.inv_freq.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> np.ndarray:
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        seq_len = int(torch.max(position_ids).item()) + 1
        if seq_len > self.max_seq_len and "dynamic" not in self.rope_type:
            raise ValueError(
                f"Sequence length {seq_len} exceeds cached max_seq_len {self.max_seq_len} for non-dynamic RoPE."
            )
        if self.position_table is None:
            self._build_position_table(device=x.device, dtype=x.dtype)
        indices = position_ids.view(-1).cpu().numpy()
        assert self.position_table is not None
        return self.position_table[indices][None, None, :, :]


class MobilintEagle3FCProjector(MobilintEagle3ModelMixin, PreTrainedModel):
    """FC projection backend used by the draft model."""

    npu_backend_prefix = "fc_"

    def project(self, hidden_states: torch.Tensor, *, count_npu_time: bool = False) -> torch.Tensor:
        """Project base hidden states to the draft hidden size.

        Args:
            hidden_states: Base hidden states, typically ``[1, seq, hidden_base]``.

        Returns:
            Projected hidden states, typically ``[1, seq, hidden_draft]``.
        """
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
    """Shared runtime helpers for an EAGLE-3 base model.

    Responsibility:
    - Convert PyTorch tensors to MXQ-compatible numpy payloads.
    - Run chunked prefill/decode against base MXQ backend.
    - Reassemble hidden/logits chunks back to torch tensors.
    """

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
        else:
            batch_size, target_length = input_shape
            combined_attention_mask = torch.zeros(
                (batch_size, 1, target_length, past_key_values_length + target_length),
                dtype=torch.float32,
                device=inputs_embeds.device,
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
        """Run base model forward with chunked MXQ inference.

        Key shapes:
        - ``inputs_embeds_numpy``: ``[1, 1, seq_chunk, hidden]``
        - ``attention_mask_numpy``: ``[1, 1, seq_chunk, cache+seq_chunk]``
        - return hidden: ``{"hidden_states": [ [1, seq, hidden] ]}``
        - return logits: ``[1, seq, vocab]`` or final-step ``[1, vocab]``
        """
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
        chunk_size = self.resolve_npu_prefill_chunk_size(self.config.eagle3_npu_chunk_size)
        timing_phase = "prefill" if past_key_values_length == 0 else "decode"
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
    """Shared runtime helpers for an EAGLE-3 draft model.

    Responsibility:
    - Consume accepted/base hidden states.
    - Expand a top-k draft tree up to configured depth.
    - Return tree metadata consumed by posterior evaluation.
    """

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
        else:
            batch_size, target_length = input_shape
            combined_attention_mask = torch.zeros(
                (batch_size, 1, target_length, past_key_values_length + target_length),
                dtype=torch.float32,
                device=hidden_states.device,
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
        """Run draft model forward with cache-aware tree attention.

        Args:
            hidden_states: ``[1, seq, hidden_base_or_draft]``.
            input_ids: ``[1, seq]`` aligned with ``hidden_states``.

        Returns:
            Tuple of:
            - hidden features (all steps or last step)
            - logits (all steps or last step)
        """
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

        chunk_size = self.resolve_npu_prefill_chunk_size(self.config.eagle3_npu_chunk_size)
        hidden_chunks: list[torch.Tensor] = []
        logits_chunks: list[torch.Tensor] = []
        base_cache_position = cache.get_draft_seq_length() + add_cache_position
        timing_phase = "prefill" if base_cache_position == 0 else "decode"
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
        logits_out = torch.cat(logits_chunks, dim=-2)
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
        """Build draft tree candidates by iterative top-k expansion.

        Returns:
            - ``draft_tokens``: ``[1, total_tokens+1]`` (root + selected nodes)
            - ``retrieve_indices``: ``[leaf_count, depth]`` paths for each leaf
            - ``tree_mask``: ``[1, 1, total_tokens+1, total_tokens+1]``
            - ``tree_position_ids``: ``[total_tokens+1]`` depth per node
        """
        total_tokens = self.max_draft_tokens if max_draft_tokens is None else min(self.max_draft_tokens, max(1, max_draft_tokens))
        depth = self.depth
        top_k = self.top_k
        sample_token = input_ids[:, -1]
        scores_list = []
        parents_list = []
        draft_token_steps = []

        input_ids_without_prompt_token = input_ids[:, 1:].to(hidden_states.device)
        draft_seq_length = cache.get_draft_seq_length()
        # `input_ids` may already be a prompt-delta sequence when `past_key_values`
        # is reused across turns. In that case, slicing by absolute draft cache
        # length would incorrectly produce an empty tensor.
        input_ids_delta = (
            input_ids_without_prompt_token[:, draft_seq_length:]
            if input_ids_without_prompt_token.shape[1] > draft_seq_length
            else input_ids_without_prompt_token
        )
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

        # The first draft depth level is already created above from
        # `last_hidden_logits` via initial top-k selection, so this loop should
        # only perform the remaining expansion rounds.
        for depth_index in range(max(0, depth - 1)):
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
        available_candidates = int(scores_list_tensor.numel())
        effective_total_tokens = min(total_tokens, available_candidates)
        if effective_total_tokens < total_tokens:
            logger.warning(
                (
                    "Requested draft tokens (%d) exceed available tree candidates (%d). "
                    "Capping to %d."
                ),
                total_tokens,
                available_candidates,
                effective_total_tokens,
            )
        top_scores = torch.topk(scores_list_tensor, effective_total_tokens, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
        draft_tokens = draft_step_tensor[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask_tensor = torch.eye(effective_total_tokens + 1).bool()
        tree_mask_tensor[:, 0] = True
        for token_index in range(effective_total_tokens):
            tree_mask_tensor[token_index + 1].add_(tree_mask_tensor[mask_index_list[token_index]])
        tree_position_ids = torch.sum(tree_mask_tensor, dim=1) - 1
        tree_mask_tensor = tree_mask_tensor.float()[None, None]
        draft_tokens = draft_tokens[None]

        max_depth = torch.max(tree_position_ids) + 1
        non_leaf_index = set(torch.unique(mask_index).tolist())
        leaf_count = effective_total_tokens - (len(non_leaf_index) - 1)
        retrieve_indices = torch.zeros(leaf_count, max_depth.item(), dtype=torch.long) - 1
        retrieve_index_rows = retrieve_indices.tolist()
        row_id = 0
        position_ids_list = tree_position_ids.tolist()
        for token_index in range(effective_total_tokens + 1):
            if token_index not in non_leaf_index:
                current_id = token_index
                node_depth = position_ids_list[token_index]
                for depth_index in reversed(range(node_depth + 1)):
                    retrieve_index_rows[row_id][depth_index] = current_id
                    current_id = mask_index_list[current_id - 1]
                row_id += 1
        if logits_processor is not None:
            max_item = effective_total_tokens + 5

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



# NOTE: Tree-decoding primitives are centralized in ``tree_decoding.py``.
# Keep re-exports here for backward compatibility with existing imports.
from .tree_decoding import (  # noqa: E402
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    softmax_topk_cpu_torch,
    tree_decoding,
    update_inference_inputs,
)
