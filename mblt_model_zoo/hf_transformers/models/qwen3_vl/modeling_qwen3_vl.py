import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, can_return_tuple, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintDeepStackCache
from ...utils.generation_utils import (
    MobilintGenerationMixin,
    build_loss_kwargs_dynamic,
    mirror_output_fields,
    pop_loss_only_kwargs,
    upstream_positional_params,
    with_mobilint_generation_signature,
)
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen3_vl import (
    MobilintQwen3VLConfig,
    MobilintQwen3VLTextConfig,
    MobilintQwen3VLVisionConfig,
)

logger = logging.get_logger(__name__)

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
except ImportError:

    @dataclass
    class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
        """Fallback Qwen3-VL vision output used by older Transformers releases."""

        deepstack_features: Optional[list[torch.FloatTensor]] = None


@lru_cache(maxsize=1)
def _upstream_qwen3_vl_uses_structured_vision_outputs() -> bool:
    """Check whether the installed Transformers expects ``visual()`` to return a model output.

    Returns:
        ``True`` when the installed upstream ``Qwen3VLModel.get_image_features`` reads structured
        fields such as ``pooler_output``. ``False`` for older releases that expect
        ``visual()`` to return ``(image_embeds, deepstack_embeds)`` directly.
    """
    get_image_features = inspect.unwrap(Qwen3VLModel.get_image_features)
    code = getattr(get_image_features, "__code__", None)
    if code is not None:
        return "pooler_output" in code.co_names

    try:
        return "pooler_output" in inspect.getsource(get_image_features)
    except OSError:
        return True


class MobilintQwen3VLPreTrainedModel(Qwen3VLPreTrainedModel):
    config: MobilintQwen3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}


class MobilintQwen3VLVisionModel(MobilintModelMixin, MobilintQwen3VLPreTrainedModel):
    config: MobilintQwen3VLVisionConfig
    input_modalities = ("image", "video")

    @classmethod
    def _from_config(cls, config: MobilintQwen3VLVisionConfig, **kwargs: Any) -> "MobilintQwen3VLVisionModel":
        """Allow Transformers AutoModel submodule construction for composite Qwen3-VL models."""
        kwargs["_internal_call"] = True
        return super()._from_config(config, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        """Expose the MXQ vision input dtype expected by upstream Qwen3-VL helpers."""
        return torch.float32

    @property
    def spatial_merge_size(self) -> int:
        """Expose the merge factor expected by upstream Qwen3-VL helpers."""
        return int(self.config.spatial_merge_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithDeepstackFeatures]:
        """Run the NPU vision encoder and adapt to the upstream Qwen3-VL vision contract.

        The compiled encoder expects a fixed-shape tensor with the following flow:

        1. HF processor output: `(256, 1536)` for a 224x224 image
        2. Runtime repreprocess: `(1, 6, 1024, 64)`
        3. Final MXQ input: `(1024, 64, 6)`

        The Mobilint backend exposes merged image embeds and deepstack features only, so
        `last_hidden_state`, `hidden_states`, and `attentions` remain unavailable.
        """
        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None and _upstream_qwen3_vl_uses_structured_vision_outputs():
            return_dict = self.config.return_dict
        del kwargs
        if hidden_states.ndim < 2:
            raise ValueError(f"Expected pixel tensor with rank >=2, got shape {tuple(hidden_states.shape)}")

        image_embeds, deepstack_embeds = self._encode_images(hidden_states, grid_thw)
        structured_outputs = BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None,
            pooler_output=image_embeds,
            hidden_states=None,
            attentions=None,
            deepstack_features=deepstack_embeds,
        )
        if return_dict is True:
            return structured_outputs
        if _upstream_qwen3_vl_uses_structured_vision_outputs():
            if return_dict is False:
                return structured_outputs.to_tuple()
            return structured_outputs
        return image_embeds, deepstack_embeds

    def _repreprocess_pixel_values(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Match the runtime `repreprocess_pixel_values` layout for one Qwen3-VL image input."""
        gt, gh, gw = grid_thw.tolist()
        c = int(self.config.in_channels)
        pt = int(self.config.temporal_patch_size)
        merge_size = int(self.config.spatial_merge_size)
        gh_merged = gh // merge_size
        gw_merged = gw // merge_size
        ph = pw = int((hidden_states.shape[-1] // (pt * c)) ** 0.5)

        expected_tokens = gt * gh_merged * gw_merged * merge_size * merge_size
        expected_hidden = c * pt * ph * pw
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                f"Unexpected pixel token count for Qwen3-VL vision input: {hidden_states.shape[0]} vs {expected_tokens}"
            )
        if hidden_states.shape[1] != expected_hidden:
            raise ValueError(
                f"Unexpected pixel hidden size for Qwen3-VL vision input: {hidden_states.shape[1]} vs {expected_hidden}"
            )

        hidden_states = hidden_states.view(
            gt,
            gh_merged,
            gw_merged,
            merge_size,
            merge_size,
            c,
            pt,
            ph,
            pw,
        )
        hidden_states = hidden_states.permute(0, 6, 5, 1, 2, 7, 3, 4, 8).contiguous()
        hidden_states = hidden_states.view(
            gt,
            pt * c,
            gh_merged * gw_merged * ph,
            merge_size * merge_size * pw,
        )
        return hidden_states

    def _prepare_npu_inputs(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> np.ndarray:
        """Convert runtime repreprocess output to the exact MXQ input shape.

        Args:
            hidden_states: HF processor pixel values, typically `(256, 1536)`.
            grid_thw: Visual grid metadata, typically `[[1, 16, 16]]`.

        Returns:
            Float32 numpy tensor with shape `(1024, 64, 6)`.
        """
        processed = self._repreprocess_pixel_values(hidden_states, grid_thw)
        if processed.ndim != 4 or processed.shape[0] != 1:
            raise ValueError(f"Unexpected preprocessed vision tensor shape: {tuple(processed.shape)}")

        # `(1, 6, 1024, 64)` -> `(1024, 64, 6)`
        processed = processed.squeeze(0).permute(1, 2, 0).contiguous()
        return processed.to(torch.float32).cpu().numpy()

    def _split_hidden_states_by_grid(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Split flattened processor image tokens according to `grid_thw` rows."""
        offset = 0
        chunks: list[torch.Tensor] = []
        for grid in grid_thw:
            gt, gh, gw = grid.tolist()
            token_count = int(gt * gh * gw)
            chunks.append(hidden_states[offset : offset + token_count])
            offset += token_count
        if offset != int(hidden_states.shape[0]):
            raise ValueError(f"Unexpected total Qwen3-VL pixel token count: {hidden_states.shape[0]} vs {offset}")
        return chunks

    def _reorder_encoder_outputs(
        self,
        encoder_outputs: list[np.ndarray],
        device: torch.device,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(encoder_outputs) < 4:
            raise ValueError(f"Expected at least 4 encoder outputs, got {len(encoder_outputs)}")

        image_embeds = self._flatten_encoder_output(encoder_outputs[0], device=device, batch_size=batch_size)
        deepstack_embeds = [
            self._flatten_encoder_output(encoder_outputs[2], device=device, batch_size=batch_size),
            self._flatten_encoder_output(encoder_outputs[3], device=device, batch_size=batch_size),
            self._flatten_encoder_output(encoder_outputs[1], device=device, batch_size=batch_size),
        ]
        return image_embeds, deepstack_embeds

    def _flatten_encoder_output(
        self,
        output: np.ndarray,
        *,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Normalize Qwen3-VL MXQ vision output to `(total_image_tokens, hidden_size)`."""
        output_array = np.asarray(output)
        if output_array.ndim >= 3 and int(output_array.shape[0]) == batch_size:
            output_array = output_array.reshape(-1, int(output_array.shape[-1]))
        else:
            output_array = np.squeeze(output_array)
            if output_array.ndim > 2:
                output_array = output_array.reshape(-1, int(output_array.shape[-1]))
        if output_array.ndim != 2:
            raise ValueError(f"Unexpected Qwen3-VL vision output shape: {tuple(np.asarray(output).shape)}")
        return torch.tensor(output_array, dtype=torch.float32, device=device)

    def _split_video_into_frames(
        self,
        chunk: torch.Tensor,
        grid: torch.Tensor,
    ) -> list[np.ndarray]:
        """Split a video chunk (gt > 1) into per-frame NPU inputs."""
        gt, gh, gw = grid.tolist()
        tokens_per_frame = int(gh * gw)
        frame_grid = torch.tensor([1, gh, gw], dtype=grid.dtype, device=grid.device)
        frames = []
        for f in range(int(gt)):
            frame_chunk = chunk[f * tokens_per_frame : (f + 1) * tokens_per_frame]
            frames.append(self._prepare_npu_inputs(frame_chunk, frame_grid))
        return frames

    def _encode_images(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run Qwen3-VL vision encoding with core-mode-specific batch handling."""
        chunks = self._split_hidden_states_by_grid(hidden_states, grid_thw)
        npu_inputs: list[np.ndarray] = []
        for chunk, grid in zip(chunks, grid_thw):
            gt = grid[0].item()
            if gt > 1:
                npu_inputs.extend(self._split_video_into_frames(chunk, grid))
            else:
                npu_inputs.append(self._prepare_npu_inputs(chunk, grid))
        npu_backend = getattr(self, "npu_backend", None)
        core_mode = getattr(npu_backend, "core_mode", getattr(self.config, "core_mode", "single"))
        mxq_model = self.get_mxq_model()
        if core_mode == "multi" and len(npu_inputs) > 1:
            encoder_outputs = mxq_model.infer(np.stack(npu_inputs, axis=0))
            if encoder_outputs is None:
                raise RuntimeError("Vision MXQ inference returned None.")
            return self._reorder_encoder_outputs(encoder_outputs, hidden_states.device, batch_size=len(npu_inputs))

        image_embeds: list[torch.Tensor] = []
        deepstack_by_layer: list[list[torch.Tensor]] = []
        for npu_input in npu_inputs:
            encoder_outputs = mxq_model.infer(npu_input)
            if encoder_outputs is None:
                raise RuntimeError("Vision MXQ inference returned None.")
            image_embed, deepstack_embeds = self._reorder_encoder_outputs(encoder_outputs, hidden_states.device)
            image_embeds.append(image_embed)
            if not deepstack_by_layer:
                deepstack_by_layer = [[] for _ in deepstack_embeds]
            for layer_idx, deepstack_embed in enumerate(deepstack_embeds):
                deepstack_by_layer[layer_idx].append(deepstack_embed)

        return torch.cat(image_embeds, dim=0), [torch.cat(layer_embeds, dim=0) for layer_embeds in deepstack_by_layer]


class MobilintQwen3VLRotaryEmbedding(nn.Module):
    """Pre-computed MRoPE for Qwen3-VL on MXQ.

    Builds a 1-D ``position_table[max_pos, peSize]`` at init (rotateTensor
    format, same as ``LlamaRotaryEmbedding`` in draftMXQ) and three
    dimension masks derived from ``mrope_section``.  At forward time the
    table is indexed by the per-dimension position ids and merged via the
    masks — no matmul, cos/sin, or interleave at runtime.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

        dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chSize = dim
        tgt_half = ((chSize + 63) // 64) * 64
        self.peSize = 2 * tgt_half

        self._build_dim_masks()
        self._build_position_table(device=device)

    def _build_dim_masks(self):
        """Build boolean masks mapping each peSize entry to T / H / W."""
        dim = self.head_dim
        halfDim = dim // 2

        freq_dim = np.full(dim // 2, 0, dtype=np.int32)  # default: T
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            indices = np.arange(offset, length, 3)
            freq_dim[indices] = dim_idx

        pe_dim = np.full(self.peSize, -1, dtype=np.int32)
        for fi in range(halfDim):
            d = freq_dim[fi]
            pe_dim[2 * fi] = d       # cos slot (first half)
            pe_dim[2 * fi + 1] = d   # -sin slot (first half)
        for fi in range(halfDim):
            d = freq_dim[fi]
            base = dim + 2 * fi
            if base < self.peSize:
                pe_dim[base] = d      # sin slot (second half)
            if base + 1 < self.peSize:
                pe_dim[base + 1] = d  # cos slot (second half)

        self.mask_t = pe_dim == 0
        self.mask_h = pe_dim == 1
        self.mask_w = pe_dim == 2

    def _build_position_table(self, device=None):
        """Pre-compute rotateTensor rows for positions 0..max_seq_len-1."""
        if device is None:
            device = self.inv_freq.device

        with torch.no_grad():
            T = self.max_seq_len
            t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)             # [T, dim]

            cos_val = emb.cos()
            sin_val = emb.sin()

            dim = self.head_dim
            halfDim = dim // 2

            cos_ = cos_val.unsqueeze(0).unsqueeze(0)  # [1, 1, T, dim]
            sin_ = sin_val.unsqueeze(0).unsqueeze(0)

            rotateTensor = torch.zeros(1, 1, T, 2 * dim, device=device, dtype=torch.float32)
            rotateTensor[..., 0:dim:2] = cos_[..., :halfDim]
            rotateTensor[..., 1:dim:2] = -sin_[..., :halfDim]
            rotateTensor[..., dim:2 * dim:2] = sin_[..., halfDim:dim]
            rotateTensor[..., dim + 1:2 * dim:2] = cos_[..., halfDim:dim]

            if rotateTensor.shape[-1] != self.peSize:
                pad = self.peSize - rotateTensor.shape[-1]
                if pad > 0:
                    rotateTensor = torch.nn.functional.pad(rotateTensor, (0, pad))

            self.position_table = rotateTensor.cpu().numpy()[0, 0]  # [T, peSize]

    @torch.no_grad()
    def forward(self, x, position_ids):
        """Index pre-computed table by 3-D position ids.

        Args:
            x: unused (API compat with upstream rotary_emb).
            position_ids: ``(3, batch, seq_len)`` or ``(batch, seq_len)``.

        Returns:
            numpy array of shape ``(batch, seq_len, peSize)``.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        pos_np = position_ids.cpu().numpy()  # (3, B, S)
        batch_size = pos_np.shape[1]
        seq_len = pos_np.shape[2]

        max_pos = int(pos_np.max()) + 1
        if max_pos > self.max_seq_len:
            self.max_seq_len = max_pos
            self._build_position_table(device=self.inv_freq.device)

        result = np.empty((batch_size, seq_len, self.peSize), dtype=np.float32)
        for b in range(batch_size):
            rows_t = self.position_table[pos_np[0, b]]  # (S, peSize)
            rows_h = self.position_table[pos_np[1, b]]
            rows_w = self.position_table[pos_np[2, b]]
            buf = result[b]
            buf[:, self.mask_t] = rows_t[:, self.mask_t]
            buf[:, self.mask_h] = rows_h[:, self.mask_h]
            buf[:, self.mask_w] = rows_w[:, self.mask_w]

        return result


class MobilintQwen3VLTextModel(MobilintModelMixin, MobilintGenerationMixin, MobilintQwen3VLPreTrainedModel):
    config: MobilintQwen3VLTextConfig
    input_modalities = ("text",)

    @classmethod
    def _from_config(cls, config: MobilintQwen3VLTextConfig, **kwargs: Any) -> "MobilintQwen3VLTextModel":
        """Allow Transformers AutoModel submodule construction for composite Qwen3-VL models."""
        kwargs["_internal_call"] = True
        return super()._from_config(config, **kwargs)

    def __init__(self, config: MobilintQwen3VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = MobilintQwen3VLRotaryEmbedding(config)
        self.num_deepstack_layers = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def _get_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        *args: object,
    ) -> MobilintDeepStackCache:
        """Return a Qwen3-VL cache that also supplies deepstack decoder chunks."""
        del cache_implementation, batch_size, max_cache_len, args
        configured_batch_size = max(1, getattr(self.config, "max_batch_size", 1))
        needs_new_cache = (
            not hasattr(self, "_cache")
            or not isinstance(self._cache, MobilintDeepStackCache)
            or getattr(self._cache, "batch_size", 1) != configured_batch_size
            or self._cache.num_deepstack_layers != self.num_deepstack_layers
            or self._cache.hidden_size != int(self.config.hidden_size)
        )
        if needs_new_cache:
            self._cache = MobilintDeepStackCache(
                self.get_cache_mxq_model(),
                batch_size=configured_batch_size,
                num_deepstack_layers=self.num_deepstack_layers,
                hidden_size=int(self.config.hidden_size),
            )
        else:
            self._cache.reset()
        return self._cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintDeepStackCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        npu_prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor,
                torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device),
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        logits = self.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            npu_prefill_chunk_size=npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            logits_to_keep=logits_to_keep,
            position_embeddings=position_embeddings,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
        )

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        deepstack_visual_embeds: Optional[list[torch.Tensor]],
        visual_pos_masks: Optional[torch.Tensor],
        past_key_values: Optional[MobilintDeepStackCache],
        cache_position: torch.Tensor,
        npu_prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 1,
        position_embeddings: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Run the dual-input MXQ decoder with HF-style ``logits_to_keep``.

        ``logits_to_keep`` follows the ``transformers`` semantics: ``1``
        (default) returns only the last-token logits; ``0`` returns every
        position; ``>1`` returns the last N positions; a ``torch.Tensor``
        selects specific positions.

        Args:
            inputs_embeds: Token embeddings of shape `(batch, seq_len, hidden)`.
            deepstack_visual_embeds: Optional deepstack features by layer.
            visual_pos_masks: Optional visual token mask.
            past_key_values: Mobilint deepstack KV cache.
            cache_position: Cache position range.
            npu_prefill_chunk_size: Optional chunk size.
            count_npu_time: Whether to accumulate NPU time.
            logits_to_keep: HF-style position selector; see the shared
                :meth:`MobilintModelMixin.llm_forward` for details.
            position_embeddings: Pre-computed RoPE numpy array of shape
                ``(batch, seq_len, peSize)`` from
                :class:`MobilintQwen3VLRotaryEmbedding`.

        Returns:
            Decoder logits for the requested token positions.
        """
        if inputs_embeds.ndim != 3:
            raise ValueError(f"Expected inputs_embeds rank 3, got shape {tuple(inputs_embeds.shape)}")
        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Mobilint Qwen3-VL currently supports batch size 1 only.")
        if past_key_values is not None and not isinstance(past_key_values, MobilintDeepStackCache):
            raise TypeError("Qwen3-VL text decoding requires MobilintDeepStackCache.")

        deepstack_tensor = self._build_deepstack_tensor(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        if past_key_values is not None:
            past_key_values.set_deepstack_tensor(deepstack_tensor)

        inputs_np = inputs_embeds.type(torch.float32).cpu().numpy()
        seq_len = int(inputs_np.shape[1])

        resolved_npu_prefill_chunk_size = self.resolve_npu_prefill_chunk_size(npu_prefill_chunk_size)

        mxq_model = self.get_mxq_model()
        self.npu_time = 0.0 if count_npu_time else None

        def _do_infer(start_index: int, end_index: int) -> np.ndarray:
            # See modeling_utils.llm_forward._do_infer: without a caller cache,
            # start_index is the running "processed so far" count within this
            # call, so use it as the KV cache_size — otherwise Path 3's prefix
            # walks would pass 0 to every chunk and the size-1 kept-position
            # captures would see no left context.
            cache_size = (
                past_key_values.get_seq_length() if past_key_values is not None else start_index
            )
            inputs_chunk = inputs_np[:, start_index:end_index, :]
            if past_key_values is None:
                deepstack_chunk = deepstack_tensor[:, start_index:end_index, :].to(dtype=torch.float32).cpu().numpy()
            else:
                deepstack_chunk = past_key_values.get_deepstack_chunk(
                    start_index,
                    end_index,
                    device=inputs_embeds.device,
                    dtype=torch.float32,
                ).cpu().numpy()

            rope_chunk = position_embeddings[:, start_index:end_index, :] if position_embeddings is not None else None
            infer_inputs = [inputs_chunk, deepstack_chunk] + ([rope_chunk] if rope_chunk is not None else [])

            if count_npu_time:
                import time

                t1 = time.perf_counter()
                result = mxq_model.infer(infer_inputs, None, cache_size)
                assert self.npu_time is not None
                self.npu_time += time.perf_counter() - t1
            else:
                result = mxq_model.infer(infer_inputs, None, cache_size)

            if result is None:
                raise RuntimeError("Text MXQ inference returned None.")
            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])
            return result[0]

        # The 3-path dispatch (fast / dynamic-axis / fallback) lives in the
        # shared helper so single-input and dual-input decoders stay in sync.
        # Unlike the single-input caller, we keep the leading batch axis
        # produced by ``do_infer``.
        return self._run_chunked_logits_to_keep(
            do_infer=_do_infer,
            seq_len=seq_len,
            npu_prefill_chunk_size=resolved_npu_prefill_chunk_size,
            logits_to_keep=logits_to_keep,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    def _build_deepstack_tensor(
        self,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Build dense deepstack input aligned to the decoder sequence.

        Args:
            inputs_embeds: Token embeddings used to infer sequence length and dtype.
            visual_pos_masks: Visual token mask from the multimodal model.
            deepstack_visual_embeds: Sparse visual embeddings per deepstack layer.

        Returns:
            Dense tensor of shape `(num_layers, seq_len, hidden_size)`.
        """
        seq_len = int(inputs_embeds.shape[1])
        hidden_size = int(inputs_embeds.shape[2])
        num_layers = self.num_deepstack_layers
        if deepstack_visual_embeds is None:
            return torch.zeros(
                (num_layers, seq_len, hidden_size),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if visual_pos_masks is None:
            raise ValueError("visual_pos_masks must be provided when deepstack_visual_embeds is not None.")

        mask = visual_pos_masks[0]
        num_layers = len(deepstack_visual_embeds)
        padded = torch.zeros((num_layers, seq_len, hidden_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for layer_idx, deepstack_embed in enumerate(deepstack_visual_embeds):
            padded[layer_idx, mask, :] = deepstack_embed.to(inputs_embeds.device, inputs_embeds.dtype)
        return padded


class MobilintQwen3VLModel(PretrainedOnlyMixin, MobilintQwen3VLPreTrainedModel, Qwen3VLModel):
    _no_split_modules = []

    def __init__(self, config: MobilintQwen3VLConfig, *args, **kwargs):
        MobilintQwen3VLPreTrainedModel.__init__(self, config, *args, **kwargs)
        self.visual = MobilintQwen3VLVisionModel._from_config(config.vision_config, _internal_call=True)
        self.language_model = MobilintQwen3VLTextModel._from_config(config.text_config, _internal_call=True)
        self.language_model.num_deepstack_layers = len(config.vision_config.deepstack_visual_indexes)
        self.rope_deltas = None


class MobilintQwen3VLForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3VLPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3VLForConditionalGeneration,
):
    def __init__(self, config: MobilintQwen3VLConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)

        self.model = MobilintQwen3VLModel(config, _internal_call=True)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()

    def get_cache_mxq_model(self):
        return self.model.language_model.get_mxq_model()

    def _get_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        *args: object,
    ) -> MobilintDeepStackCache:
        """Delegate generation cache creation to the Qwen3-VL language model."""
        return self.model.language_model._get_cache(cache_implementation, batch_size, max_cache_len, *args)

    @with_mobilint_generation_signature(
        Qwen3VLForConditionalGeneration.prepare_inputs_for_generation,
        "count_npu_time",
        "npu_prefill_chunk_size",
    )
    def prepare_inputs_for_generation(
        self,
        *args: Any,
        count_npu_time: bool = False,
        npu_prefill_chunk_size: int | None = None,
        **kwargs: Any,
    ):
        """Prepare generation inputs while preserving Mobilint timing kwargs.

        Args:
            *args: Positional arguments forwarded to the upstream Qwen3-VL generation helper.
            count_npu_time: Whether Mobilint decoder NPU time should be accumulated.
            npu_prefill_chunk_size: Optional prefill chunk size forwarded to Mobilint generation.
            **kwargs: Keyword arguments forwarded to the upstream Qwen3-VL generation helper.

        Returns:
            Model inputs for a generation step.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["count_npu_time"] = count_npu_time
        if npu_prefill_chunk_size is not None:
            model_inputs["npu_prefill_chunk_size"] = npu_prefill_chunk_size
        return model_inputs

    @with_mobilint_generation_signature(Qwen3VLForConditionalGeneration.forward, "count_npu_time")
    @can_return_tuple
    def forward(
        self,
        *args: Any,
        count_npu_time: bool = False,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        """Route ``logits_to_keep`` to the Mobilint text decoder.

        Upstream ``Qwen3VLForConditionalGeneration.forward`` extracts ``logits_to_keep``
        as a named argument and performs its own final slice on the text model output,
        which bypasses the Mobilint decoder's position selection. To keep that decoder
        in charge of picking positions, we pop ``logits_to_keep`` here and thread it
        into ``self.model`` via kwargs (upstream ``Qwen3VLModel.forward`` forwards its
        own ``**kwargs`` to the text model). All other arguments follow the upstream
        signature by way of ``@with_mobilint_generation_signature``, so upstream
        additions such as ``mm_token_type_ids`` continue to pass through unchanged.

        Tuple mode: ``@can_return_tuple`` strips ``return_dict`` from kwargs before
        the wrapper body runs (so ``self.model`` never returns a tuple) and converts
        the assembled ``Qwen3VLCausalLMOutputWithPast`` back to a tuple when
        ``return_dict=False`` was requested — matching the upstream forward's
        contract.

        Dynamic adaptation:
            * Loss kwargs are built via :func:`build_loss_kwargs_dynamic`, so
              upstream additions like ``num_items_in_batch`` / ``shift_labels``
              flow through when the loss function accepts them.
            * The returned ``Qwen3VLCausalLMOutputWithPast`` is assembled by
              :func:`mirror_output_fields`, so new output fields (e.g. a future
              ``image_hidden_states``) are mirrored from the upstream model
              output automatically instead of requiring wrapper edits.

        Performance: the default ``logits_to_keep=0`` (keep-all) matches HF but on
        last-only MXQ triggers a size-1 infer per input token. ``.generate()`` is
        safe (HF passes ``logits_to_keep=1``); manual ``.forward()`` callers doing
        perplexity eval / logit collection inherit this cost on last-only builds.
        """
        positional_params = upstream_positional_params(Qwen3VLForConditionalGeneration.forward)
        if len(args) > len(positional_params):
            raise TypeError(
                f"forward() takes at most {len(positional_params)} positional arguments "
                f"but {len(args)} were given"
            )
        for name, value in zip(positional_params, args):
            if name in kwargs:
                raise TypeError(f"forward() got multiple values for argument {name!r}")
            kwargs[name] = value

        labels = kwargs.pop("labels", None)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        # Loss-only kwargs (``num_items_in_batch``, ``shift_labels``) must be
        # stripped BEFORE ``self.model`` is called so they don't reach the
        # inner text model via upstream ``Qwen3VLModel``'s ``**kwargs`` pass-
        # through. Keeps parity with the Qwen2-VL wrapper.
        loss_only_kwargs = pop_loss_only_kwargs(kwargs)

        outputs = self.model(
            logits_to_keep=logits_to_keep,
            count_npu_time=count_npu_time,
            **kwargs,
        )

        # The Mobilint text decoder already returns logits sliced to the requested
        # positions and ``self.lm_head`` is ``nn.Identity``, so skip the upstream
        # ``hidden_states[:, slice_indices, :]`` step.
        logits = cast(torch.FloatTensor, self.lm_head(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                **build_loss_kwargs_dynamic(
                    self.loss_function,
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.text_config.vocab_size,
                    upstream_kwargs=loss_only_kwargs,
                )
            )

        return mirror_output_fields(
            Qwen3VLCausalLMOutputWithPast,
            outputs,
            loss=loss,
            logits=logits,
        )


AutoModel.register(MobilintQwen3VLVisionConfig, MobilintQwen3VLVisionModel)
AutoModel.register(MobilintQwen3VLTextConfig, MobilintQwen3VLTextModel)
AutoModel.register(MobilintQwen3VLConfig, MobilintQwen3VLForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintQwen3VLConfig, MobilintQwen3VLForConditionalGeneration)
