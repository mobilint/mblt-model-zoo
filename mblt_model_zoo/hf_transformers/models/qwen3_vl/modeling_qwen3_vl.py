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
    Qwen3VLVisionRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, can_return_tuple, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import (
    MobilintDeepStackCache,
    build_mobilint_cache_from_model,
    cache_matches_backend_topology,
)
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


@lru_cache(maxsize=1)
def _upstream_vision_rotary_takes_position_ids() -> bool:
    """Return True when upstream ``Qwen3VLVisionRotaryEmbedding.forward`` takes ``position_ids``.

    Older Transformers releases ship ``forward(self, seqlen: int)`` — we can
    pass the max HW extent directly and index the returned freq table with
    2-D coordinates. Newer releases (transformers 5.x onward) switched to
    ``forward(self, position_ids: torch.Tensor)`` and return the
    already-flattened freqs, so we must build the arange tensor ourselves.
    Detecting by the first non-``self`` parameter name keeps us compatible
    across the whole supported range (>=4.57.0, <=5.12.1) without pinning
    to a specific transformers version.
    """
    try:
        sig = inspect.signature(Qwen3VLVisionRotaryEmbedding.forward)
    except (TypeError, ValueError):
        return True
    params = list(sig.parameters.values())
    first = params[1].name if len(params) >= 2 else ""
    return first == "position_ids"


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


def fold_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Fused repreprocess + fold: HF pixel_values (N, fold_in) -> (1, fold_in, 1, N)."""
    n, fold_in = pixel_values.shape
    return pixel_values.transpose(0, 1).reshape(1, fold_in, 1, n).contiguous()


class MobilintQwen3VLVisionModel(MobilintModelMixin, MobilintQwen3VLPreTrainedModel):
    config: MobilintQwen3VLVisionConfig
    input_modalities = ("image", "video")

    def __init__(self, config: MobilintQwen3VLVisionConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        num_mxq_inputs = len(self.get_mxq_model().get_input_buffer_info())
        self._uses_dynamic_vision = self._resolve_dynamic_vision_flag(num_mxq_inputs)
        # Only the dynamic path consumes `pos_embed` and `rotary_pos_emb`
        # (via `_prepare_dynamic_npu_inputs`), and static Qwen3-VL Hub
        # checkpoints don't ship `visual.pos_embed.weight`. Skipping the
        # allocation on static builds avoids a spurious "MISSING: newly
        # initialized" load warning for a weight that would never be used.
        if self._uses_dynamic_vision:
            self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
            self.num_grid_per_side = int(config.num_position_embeddings**0.5)
            head_dim = config.hidden_size // config.num_heads
            self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

    @classmethod
    def _from_config(cls, config: MobilintQwen3VLVisionConfig, **kwargs: Any) -> "MobilintQwen3VLVisionModel":
        """Allow Transformers AutoModel submodule construction for composite Qwen3-VL models."""
        kwargs["_internal_call"] = True
        return super()._from_config(config, **kwargs)

    @staticmethod
    def _resolve_dynamic_vision_flag(num_mxq_inputs: int) -> bool:
        """Detect the vision dispatch path from the compiled MXQ input count.

        1-input builds take a single folded pixel tensor (static path);
        3-input builds take ``[rope, pos, folded]`` (dynamic path). Any other
        signature is a compile-side mismatch we cannot recover from — raise
        rather than guess so a wrong-shape input never reaches the NPU. The
        top-level ``config.dynamic_vision`` hint is reconciled against this
        detected value at the composite-model level (see
        ``MobilintQwen3VLModel._reconcile_dynamic_vision``); this helper stays
        purely a function of the compiled MXQ.
        """
        if num_mxq_inputs == 1:
            return False
        if num_mxq_inputs == 3:
            return True
        raise ValueError(
            f"Qwen3-VL vision MXQ must expose 1 (static) or 3 (dynamic "
            f"[rope, pos, folded]) inputs; got {num_mxq_inputs}."
        )

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

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = int(self.config.spatial_merge_size)
        max_hw = int(grid_thw[:, 1:].max().item())
        if _upstream_vision_rotary_takes_position_ids():
            inv_freq = self.rotary_pos_emb.inv_freq
            position_ids = torch.arange(max_hw, device=inv_freq.device, dtype=inv_freq.dtype)
            freq_table = self.rotary_pos_emb(position_ids)
        else:
            freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list: list[list] = [[] for _ in range(4)]
        weight_list: list[list] = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h))
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w))
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        device = self.pos_embed.weight.device
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])
        merge_size = int(self.config.spatial_merge_size)
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(int(t), 1)
            pos_embed = (
                pos_embed.view(int(t), int(h) // merge_size, merge_size, int(w) // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    @torch.no_grad()
    def compute_side_inputs(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        rotary = self._rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos_sin = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return pos_embeds, cos_sin

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

    def _prepare_dynamic_npu_inputs(
        self, hidden_states: torch.Tensor, grid: torch.Tensor
    ) -> list[np.ndarray]:
        grid_thw = grid.unsqueeze(0) if grid.dim() == 1 else grid
        folded = fold_pixel_values(hidden_states)
        pos_embeds, _ = self.compute_side_inputs(grid_thw)
        n = folded.shape[-1]

        folded_np = folded.squeeze(2).permute(0, 2, 1).contiguous().to(torch.float32).cpu().numpy()
        pos_np = pos_embeds.reshape(1, n, -1).to(torch.float32).cpu().numpy()
        rope_np = self._build_vision_rotate_tensor(grid_thw)

        return [rope_np, pos_np, folded_np]

    def _build_vision_rotate_tensor(self, grid_thw: torch.Tensor) -> np.ndarray:
        """Build rotateTensor-format rotary for the vision encoder (matches MXQ peSize layout)."""
        rotary = self._rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos_val = emb.cos()
        sin_val = emb.sin()

        n = emb.shape[0]
        dim = emb.shape[-1]
        half_dim = dim // 2
        tgt_half = ((dim + 63) // 64) * 64
        pe_size = 2 * tgt_half

        rt = torch.zeros(n, pe_size, dtype=torch.float32)
        rt[:, 0:dim:2] = cos_val[:, :half_dim]
        rt[:, 1:dim:2] = -sin_val[:, :half_dim]
        rt[:, dim : 2 * dim : 2] = sin_val[:, half_dim:]
        rt[:, dim + 1 : 2 * dim : 2] = cos_val[:, half_dim:]

        return rt.reshape(1, n, pe_size).numpy()

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

    def _split_video_into_dynamic_frames(
        self, chunk: torch.Tensor, grid: torch.Tensor
    ) -> list[list[np.ndarray]]:
        gt, gh, gw = (int(x) for x in grid.tolist())
        tokens_per_frame = gh * gw
        frame_grid = torch.tensor([1, gh, gw], dtype=grid.dtype, device=grid.device)
        frames = []
        for f in range(gt):
            frame_chunk = chunk[f * tokens_per_frame : (f + 1) * tokens_per_frame]
            frames.append(self._prepare_dynamic_npu_inputs(frame_chunk, frame_grid))
        return frames

    def _encode_images(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run Qwen3-VL vision encoding with core-mode-specific batch handling."""
        chunks = self._split_hidden_states_by_grid(hidden_states, grid_thw)
        is_dynamic = self._uses_dynamic_vision

        # Defense-in-depth video guard behind the processor-level check in
        # `MobilintQwen3VLProcessor.__call__`: static Qwen3-VL MXQ releases bake a
        # single image's 2D RoPE + fixed visual-token count into the text decoder,
        # so video (per-frame RoPE + variable visual-token count) silently degrades
        # into embeddings the language model still decodes into plausible-looking
        # but semantically wrong text. `grid_thw[i, 0] > 1` is a per-row property
        # (frame count in one video), so it is safe to enforce here — unlike
        # multi-image, which the model cannot distinguish from batched single-image
        # samples given only `grid_thw` (that guard lives in the processor).
        if not is_dynamic and any(int(g[0].item()) > 1 for g in grid_thw):
            raise NotImplementedError(
                "Video input requires a dynamic-vision Qwen3-VL MXQ (3-input vision + "
                "variable visual-token count in the text decoder). The currently loaded "
                "vision MXQ is static (single-tensor input with fixed frame size). Use a "
                "Qwen3-VL release that ships a dynamic vision MXQ, or pass only image "
                "inputs."
            )

        npu_inputs: list = []
        for chunk, grid in zip(chunks, grid_thw):
            gt = grid[0].item()
            if gt > 1:
                npu_inputs.extend(self._split_video_into_dynamic_frames(chunk, grid))
            else:
                if is_dynamic:
                    npu_inputs.append(self._prepare_dynamic_npu_inputs(chunk, grid))
                else:
                    npu_inputs.append(self._prepare_npu_inputs(chunk, grid))

        npu_backend = getattr(self, "npu_backend", None)
        core_mode = getattr(npu_backend, "core_mode", getattr(self.config, "core_mode", "single"))
        mxq_model = self.get_mxq_model()
        for i, inp in enumerate(npu_inputs):
            if isinstance(inp, list):
                shapes = [np.asarray(x).shape for x in inp]
                logger.debug("[Vision] Input[%d] (dynamic, %d tensors): %s", i, len(inp), shapes)
            else:
                logger.debug("[Vision] Input[%d] shape: %s", i, np.asarray(inp).shape)
        if not is_dynamic and core_mode == "multi" and len(npu_inputs) > 1:
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
    format, same layout as ``CachedRotaryEmbedding`` in
    ``mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils``) and three
    dimension masks derived from ``mrope_section``.  At forward time the
    table is indexed by the per-dimension position ids and merged via the
    masks — no matmul, cos/sin, or interleave at runtime.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_position_embeddings

        # Transformers 5.x folds rope_theta into `rope_parameters` (exposed via
        # the `rope_scaling` property) during config __post_init__, so the flat
        # attribute is dropped. Transformers 4.x keeps `rope_theta` as its own
        # attribute. Read both.
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None or "mrope_section" not in rope_scaling:
            raise ValueError(
                "MobilintQwen3VLRotaryEmbedding requires config.rope_scaling.mrope_section; "
                "check that the Qwen3-VL text config was loaded correctly."
            )
        self.mrope_section = rope_scaling["mrope_section"]

        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            raise ValueError(
                "MobilintQwen3VLRotaryEmbedding requires config.rope_theta (Transformers <5) or "
                "config.rope_scaling['rope_theta'] (Transformers >=5); check that the Qwen3-VL "
                "text config was loaded correctly."
            )
        self.rope_theta = rope_theta

        dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chSize = dim
        tgt_half = ((chSize + 63) // 64) * 64
        self.peSize = 2 * tgt_half

        self._build_dim_masks()
        # HF Transformers 5.x materializes weights lazily under
        # `torch.set_default_device("meta")`, so `inv_freq` starts on meta and
        # `_build_position_table` would fail at `.cpu().numpy()`. Defer the
        # table build until forward, mirroring
        # `mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils.CachedRotaryEmbedding`.
        self.position_table = None
        if self.inv_freq.device.type != "meta":
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
            dim = self.head_dim
            # Recompute inv_freq locally. On tf 5.x, `from_pretrained` loads the
            # module under `torch.set_default_device("meta")` so the `arange(...)
            # / dim` in `__init__` runs on meta and the ``inv_freq`` register_buffer
            # is later materialized off meta with uninitialized bytes (garbage
            # floats, not the correct 1 / theta^(2i/d)). Since inv_freq is a pure
            # function of (rope_theta, head_dim), recomputing here avoids the meta
            # trap and matches the tf 4.x path bit-for-bit.
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            T = self.max_seq_len
            t = torch.arange(T, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
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
        if self.position_table is None or max_pos > self.max_seq_len:
            self.max_seq_len = max(max_pos, self.max_seq_len)
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

    # Qwen3-VL text MXQ is compiled with rank-3 inputs:
    # ``(1, -1, hidden)`` for inputs_embeds and ``(num_layers, -1, hidden)``
    # for deepstack. The shared batched helper must not add the extra
    # ``expand_dims(axis=1)`` it uses for LLM-style ``(1, 1, seq, hidden)``.
    _batched_input_expand_dims = False

    @classmethod
    def _from_config(cls, config: MobilintQwen3VLTextConfig, **kwargs: Any) -> "MobilintQwen3VLTextModel":
        """Allow Transformers AutoModel submodule construction for composite Qwen3-VL models."""
        kwargs["_internal_call"] = True
        return super()._from_config(config, **kwargs)

    def __init__(self, config: MobilintQwen3VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Supported compiled layouts (detected from the MXQ variant handle,
        # not ``max_batch_size``):
        #   * Non-batch (``max_batch_size == 1``): 2 or 3 inputs.
        #     - 2-input ``[inputs_embeds (1,-1,H), deepstack (num_layers,-1,H)]``
        #       — legacy/static: MRoPE is baked into the compiled model, no
        #       external rope tensor is fed. Used by the 2B/4B W8 builds.
        #     - 3-input ``[inputs_embeds (1,-1,H), deepstack (num_layers,-1,H),
        #       rope (1,-1,peSize)]`` — dynamic: rope table produced by
        #       :class:`MobilintQwen3VLRotaryEmbedding` and threaded through
        #       ``_do_infer``. Used by the 8B W8 build shipped on HF Hub.
        #   * Batch (``max_batch_size > 1``, e.g. the Batch16 W8 build):
        #     3-input ``[inputs_embeds (1,-1,H), rope (1,-1,peSize),
        #     deepstack (num_layers,-1,H)]`` only — the legacy 2-input batch
        #     MXQ is no longer supported (see ``_llm_forward_batch_deepstack``).
        # The 3-input orders differ between non-batch and batch builds; the
        # compiled signatures are independent and each dispatch honors its own
        # layout. We trust the compiled MXQ over any config attr for the input
        # count since batch builds fuse every tensor into a single
        # ``get_input_buffer_info()`` entry (would misreport as 1). The variant
        # handle's ``get_model_input_shape()`` returns one shape per tensor
        # input regardless of buffer fusion.
        num_mxq_inputs = self._get_num_mxq_inputs()
        if num_mxq_inputs == 3:
            self._uses_rope_input = True
            self.rotary_emb: Optional[MobilintQwen3VLRotaryEmbedding] = MobilintQwen3VLRotaryEmbedding(config)
        elif num_mxq_inputs == 2:
            self._uses_rope_input = False
            self.rotary_emb = None
        else:
            raise ValueError(
                f"Qwen3-VL text MXQ must expose 2 (non-batch: [inputs, deepstack]) or "
                f"3 (non-batch: [inputs, deepstack, rope] / batch: [inputs, rope, "
                f"deepstack]) inputs; got {num_mxq_inputs}."
            )
        self.num_deepstack_layers = 0

    def _get_num_mxq_inputs(self) -> int:
        """Return the compiled MXQ's true input tensor count.

        Reads the variant handle's ``get_model_input_shape()`` rather than
        ``get_input_buffer_info()`` because batch builds fuse every input
        tensor into a single buffer-info entry (misreporting as 1). The
        variant handle exposes one shape per tensor input for both batch
        and non-batch layouts.
        """
        handle = self.get_mxq_model().get_model_variant_handle(0)
        return len(handle.get_model_input_shape())

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @classmethod
    def get_mobilint_cache_cls(cls) -> type[MobilintDeepStackCache]:
        """Qwen3-VL text decoder requires the deepstack-augmented KV cache.

        :meth:`llm_forward` hard-fails on any ``past_key_values`` that is not
        a :class:`MobilintDeepStackCache`. The default multi-slot builder in
        :mod:`benchmark_utils` reads this override to construct the deepstack
        cache directly rather than the plain :class:`MobilintCache`, so
        fake-prefill VLM decode measurements do not trip that guard on
        multi-slot backends.
        """
        return MobilintDeepStackCache

    def _get_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        *args: object,
    ) -> MobilintDeepStackCache:
        """Return a Qwen3-VL cache that also supplies deepstack decoder chunks.

        Delegates cache construction to :func:`build_mobilint_cache_from_model` so a
        multi-slot text backend (``N`` Model slots × ``K`` per-model cache IDs) routes
        each flat row to its owning ``qbruntime.Model``. The legacy single-Model path
        still falls back to ``MobilintDeepStackCache(slot_0_model, batch_size=B)``
        through the same helper.
        """
        del cache_implementation, batch_size, max_cache_len, args
        configured_batch_size = max(1, int(getattr(self.config, "max_batch_size", 1)))
        existing_cache = getattr(self, "_cache", None)
        needs_new_cache = (
            not isinstance(existing_cache, MobilintDeepStackCache)
            or getattr(existing_cache, "batch_size", 1) < configured_batch_size
            or existing_cache.num_deepstack_layers != self.num_deepstack_layers
            or existing_cache.hidden_size != int(self.config.hidden_size)
            # A dispose+relaunch of the text backend can swap the (mxq_models,
            # k_per_model) topology while preserving the aggregate row capacity;
            # the cached slot routing must be rebuilt from the current slots.
            or not cache_matches_backend_topology(existing_cache, self)
        )
        if needs_new_cache:
            self._cache = build_mobilint_cache_from_model(
                self,
                configured_batch_size,
                cache_cls=MobilintDeepStackCache,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        # Route attention_mask through so batch>1 hits the batched deepstack path.
        # Mirrors the plain-LLM `resolve_batched_attention_mask` convention.
        effective_attention_mask = self.resolve_batched_attention_mask(inputs_embeds, attention_mask)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor,
                torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device),
            )

        if self._uses_rope_input:
            # the hard coded `3` is for temporal, height and width.
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:]

            assert self.rotary_emb is not None
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        else:
            position_embeddings = None

        logits = self.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            npu_prefill_chunk_size=npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            attention_mask=effective_attention_mask,
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
        attention_mask: Optional[torch.Tensor] = None,
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
            attention_mask: Batched attention mask. When provided, dispatches to
                :meth:`_llm_forward_batch_deepstack` so the compiled batched text
                MXQ can process every batch row in a single infer call.
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
        if past_key_values is not None and not isinstance(past_key_values, MobilintDeepStackCache):
            raise TypeError("Qwen3-VL text decoding requires MobilintDeepStackCache.")

        # Reset the NPU timing accumulator before either dispatch so the
        # batched path's `_run_batch_infer` (in the shared helper) does not
        # trip its `self.npu_time is not None` assertion. Base LLM does the
        # same reset up front — mirror that here so the batched deepstack
        # path is symmetric with the single-batch fallback below.
        self.npu_time = 0.0 if count_npu_time else None

        if attention_mask is not None:
            self._validate_batch_cache(past_key_values, attention_mask.shape[0])
            return self._llm_forward_batch_deepstack(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                deepstack_visual_embeds=deepstack_visual_embeds,
                visual_pos_masks=visual_pos_masks,
                past_key_values=past_key_values,
                cache_position=cache_position,
                npu_prefill_chunk_size=npu_prefill_chunk_size,
                count_npu_time=count_npu_time,
                logits_to_keep=logits_to_keep,
                position_embeddings=position_embeddings,
            )

        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError(
                "Mobilint Qwen3-VL batch>1 without attention_mask is not supported; "
                "pass an attention_mask (or configure max_batch_size>1) to use the "
                "batched deepstack path."
            )

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

            # Non-batch (``max_batch_size == 1``) builds ship two MXQ layouts:
            #   * 2-input ``[inputs, deepstack]`` — legacy/static: MRoPE baked
            #     into the compiled model, no rope tensor is fed.
            #   * 3-input ``[inputs, deepstack, rope]`` — dynamic: rope threaded
            #     externally as ``(1, seq, peSize)`` after deepstack. Note this
            #     order differs from the batched build's ``[inputs, rope,
            #     deepstack]`` (see ``_llm_forward_batch_deepstack``); the
            #     compiled signatures are independent so we honor each path's
            #     actual input layout rather than unifying them.
            infer_inputs = [inputs_chunk, deepstack_chunk]
            if self._uses_rope_input:
                assert position_embeddings is not None, (
                    "position_embeddings must be provided for the 3-input Qwen3-VL text MXQ."
                )
                infer_inputs.append(position_embeddings[:, start_index:end_index, :])

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

    def _build_batched_deepstack_tensors(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Build per-batch-item deepstack tensors sliced to their active length.

        Upstream Qwen3-VL packs deepstack embeds across the whole batch:
        ``visual_pos_masks`` is ``(batch, seq_len)`` bool and each layer's
        ``deepstack_visual_embeds`` is ``(sum(visual_pos_masks), hidden)`` in
        batch-major, sequence-minor row order. We split them per item so the
        batched infer can concat per-item chunks along the token axis the
        same way :meth:`MobilintModelMixin._assemble_batch_chunk` slices
        ``inputs_embeds_masked``.

        Returns:
            List of length ``batch_size``. Item ``j`` has shape
            ``(num_layers, seq_len_j, hidden_size)`` where ``seq_len_j`` is
            the number of non-padding tokens in row ``j`` (or 1 in the
            decode-shaped single-token path).
        """
        batch_size = int(inputs_embeds.shape[0])
        hidden_size = int(inputs_embeds.shape[2])
        num_layers = self.num_deepstack_layers

        if attention_mask.shape == inputs_embeds.shape[:-1]:
            attention_mask_bool = attention_mask.type(torch.bool)
            sequence_lengths = [int(attention_mask_bool[j].sum()) for j in range(batch_size)]
        else:
            # Mirrors the decode-shaped fallback in _llm_forward_batch: no
            # per-token attention mask, single-token rows.
            assert inputs_embeds.shape[1] == 1
            attention_mask_bool = None
            sequence_lengths = [1 for _ in range(batch_size)]

        if deepstack_visual_embeds is None:
            # Decode step (new tokens are never visual tokens) or a caller
            # that skips deepstack for this forward — zero contribution.
            return [
                torch.zeros(
                    (num_layers, sequence_lengths[j], hidden_size),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )
                for j in range(batch_size)
            ]

        if visual_pos_masks is None:
            raise ValueError("visual_pos_masks must be provided when deepstack_visual_embeds is not None.")

        # Trust deepstack layer count from the caller if the model attribute
        # was not set (mirrors the single-batch branch which does the same).
        effective_num_layers = num_layers if num_layers > 0 else len(deepstack_visual_embeds)

        visual_pos_masks_bool = visual_pos_masks.to(torch.bool)
        counts_per_item = [int(visual_pos_masks_bool[j].sum()) for j in range(batch_size)]
        offsets = [0]
        for count in counts_per_item[:-1]:
            offsets.append(offsets[-1] + count)

        per_item: list[torch.Tensor] = []
        for j in range(batch_size):
            seq_len_j = sequence_lengths[j]
            padded = torch.zeros(
                (effective_num_layers, seq_len_j, hidden_size),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            if attention_mask_bool is not None:
                # Restrict the visual mask to the item's active window so
                # scatter indices align with the compacted per-item embeds.
                active_mask = attention_mask_bool[j]
                visual_mask_j = visual_pos_masks_bool[j][active_mask]
            else:
                visual_mask_j = visual_pos_masks_bool[j]
            count_j = counts_per_item[j]
            start = offsets[j]
            for layer_idx, deepstack_embed in enumerate(deepstack_visual_embeds):
                if count_j == 0:
                    continue
                padded[layer_idx, visual_mask_j, :] = deepstack_embed[start : start + count_j].to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
            per_item.append(padded)
        return per_item

    def _build_batched_rope_arrays(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: np.ndarray,
    ) -> list[np.ndarray]:
        """Slice a batched ``(batch, seq_len, peSize)`` rope array per active item.

        Mirrors :meth:`_build_batched_deepstack_tensors`'s per-item slicing so
        the closure in :meth:`_llm_forward_batch_deepstack` can select the
        same ``[chunk_start, chunk_start + chunk_len_k)`` window on rope that
        ``_assemble_batch_chunk`` selects on ``inputs_embeds_masked[j]``.

        Returns:
            List of length ``batch_size``. Item ``j`` has shape
            ``(seq_len_j, peSize)`` where ``seq_len_j`` matches the compacted
            per-item embed length (or 1 in the decode-shaped fallback).
        """
        batch_size = int(inputs_embeds.shape[0])
        if attention_mask.shape == inputs_embeds.shape[:-1]:
            attention_mask_bool_np = attention_mask.type(torch.bool).cpu().numpy()
            return [position_embeddings[j, attention_mask_bool_np[j], :] for j in range(batch_size)]
        # Decode-shaped fallback: no per-token attention mask, single-token rows.
        assert inputs_embeds.shape[1] == 1
        return [position_embeddings[j, :, :] for j in range(batch_size)]

    def _llm_forward_batch_deepstack(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        deepstack_visual_embeds: Optional[list[torch.Tensor]],
        visual_pos_masks: Optional[torch.Tensor],
        past_key_values: Optional[MobilintDeepStackCache],
        cache_position: torch.Tensor,
        npu_prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 1,
        position_embeddings: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Batched sibling of :meth:`llm_forward` that also packs deepstack chunks.

        Reuses the shared 3-path dispatch in
        :meth:`MobilintModelMixin._llm_forward_batch` and supplies a
        ``pack_extra_inputs`` hook that slices per-item deepstack tensors
        with the same ``[start, start + chunk_len_k)`` windows the base
        helper uses for ``inputs_embeds_masked``. The single-input LLM path
        already handles KV cache tracking across the batch via
        ``update_seen_tokens`` — the shared helper does that here too, so
        we skip ``cache.set_deepstack_tensor`` / ``update_cache_position``
        (they are only needed by the single-batch decode replay path).

        The batched path only supports the 3-input ``[inputs, rope,
        deepstack]`` MXQ signature (i.e. the current Batch16 W8 build). The
        caller supplies a shared ``position_embeddings`` array of shape
        ``(batch, seq_len, peSize)`` pre-computed once in :meth:`forward`;
        per-item rope rows are sliced the same way as deepstack and
        concatenated so the packed extras arrive at the compiled model as
        ``[rope, deepstack]`` — the positions the 3-input MXQ expects.
        """
        del cache_position  # Batched path uses `update_seen_tokens` bookkeeping.

        if not self._uses_rope_input:
            raise ValueError(
                "Batched Qwen3-VL text inference requires a 3-input "
                "[inputs, rope, deepstack] MXQ (the current Batch16 build). "
                "The legacy 2-input batch MXQ is no longer supported."
            )
        assert position_embeddings is not None, (
            "position_embeddings must be provided for the 3-input Qwen3-VL text MXQ."
        )

        resolved_npu_prefill_chunk_size = self.resolve_npu_prefill_chunk_size(npu_prefill_chunk_size)

        deepstack_by_item = self._build_batched_deepstack_tensors(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        rope_by_item = self._build_batched_rope_arrays(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        def _pack_deepstack_extras(
            *,
            chunk_start: int,
            sequence_lengths_chunks: list[int],
            cache_ids: list[int],
        ) -> list[np.ndarray]:
            # Batched builds ship the 3-input signature
            # ``[inputs, rope, deepstack]``. Slice each active item's
            # per-item tensors with the same
            # ``[chunk_start, chunk_start + chunk_len_k)`` window the base
            # helper uses for ``inputs_embeds_masked`` and concat along the
            # token axis so the packed shapes match the compiled model:
            # rope becomes ``(1, packed_tokens, peSize)`` (leading batch
            # axis lifted from the per-item ``(seq_len_j, peSize)`` slices)
            # and deepstack becomes ``(num_layers, packed_tokens, hidden)``.
            rope_chunks: list[np.ndarray] = []
            deepstack_chunks: list[torch.Tensor] = []
            for k, cache_id in enumerate(cache_ids):
                length = sequence_lengths_chunks[k]
                end = chunk_start + length
                rope_chunks.append(rope_by_item[cache_id][chunk_start:end, :])
                deepstack_chunks.append(deepstack_by_item[cache_id][:, chunk_start:end, :])
            rope_concat = np.concatenate(rope_chunks, axis=0)[np.newaxis, :, :]
            deepstack_concat = torch.cat(deepstack_chunks, dim=1)
            return [
                rope_concat.astype(np.float32, copy=False),
                deepstack_concat.to(dtype=torch.float32).cpu().numpy(),
            ]

        return self._llm_forward_batch(
            inputs_embeds,
            attention_mask,
            past_key_values,
            resolved_npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            logits_to_keep=logits_to_keep,
            pack_extra_inputs=_pack_deepstack_extras,
        )


class MobilintQwen3VLModel(PretrainedOnlyMixin, MobilintQwen3VLPreTrainedModel, Qwen3VLModel):
    _no_split_modules = []

    def __init__(self, config: MobilintQwen3VLConfig, *args, **kwargs):
        MobilintQwen3VLPreTrainedModel.__init__(self, config, *args, **kwargs)
        self.visual = MobilintQwen3VLVisionModel._from_config(config.vision_config, _internal_call=True)
        self.language_model = MobilintQwen3VLTextModel._from_config(config.text_config, _internal_call=True)
        self.language_model.num_deepstack_layers = len(config.vision_config.deepstack_visual_indexes)
        self.rope_deltas = None
        self._reconcile_dynamic_vision(config, visual=self.visual, language_model=self.language_model)

    @staticmethod
    def _reconcile_dynamic_vision(
        config: MobilintQwen3VLConfig,
        vision_dynamic: bool | None = None,
        text_dynamic: bool | None = None,
        *,
        visual=None,
        language_model=None,
    ) -> bool:
        """Reconcile ``config.dynamic_vision`` against paired vision/text MXQ signatures.

        ``dynamic_vision`` is a release-level attribute: the vision MXQ, text
        MXQ, image processor, and video processor are a *bundled release* and
        cannot be swapped independently. A dynamic-vision MXQ produces per-image
        / per-frame RoPE tensors that the text MXQ must consume via its rope
        input; a static-vision MXQ bakes MRoPE into the text decoder. Pairing
        a dynamic vision MXQ with a legacy static text MXQ (or vice versa) is
        silently semantically wrong — the language model loses image-boundary
        information and emits plausible-but-corrupted output.

        Each submodule detects its own signature from its compiled MXQ:

        * ``visual._uses_dynamic_vision`` is True for a 3-input vision MXQ.
        * ``language_model._uses_rope_input`` is True for a 3-input text MXQ
          that receives a per-image rope tensor.

        The two flags must agree. When they disagree we raise ``ValueError``
        with both flags and both MXQ paths so the caller can either load a
        consistent release or override both sides. When they agree we promote
        the value onto ``config.dynamic_vision`` and warn once when the shipped
        config hint disagrees, so downstream consumers (processor, video
        processor) can trust the top-level attr.

        Args:
            config: Composite Qwen3-VL config carrying the top-level hint.
            vision_dynamic: Detected vision path. When ``None``, read from
                ``visual``.
            text_dynamic: Detected text path. When ``None``, read from
                ``language_model``.
            visual: Vision submodule exposing ``_uses_dynamic_vision``,
                consulted only when ``vision_dynamic`` is not supplied.
            language_model: Text submodule exposing ``_uses_rope_input``,
                consulted only when ``text_dynamic`` is not supplied.

        Returns:
            The reconciled dynamic-vision flag now stored on ``config``.

        Raises:
            ValueError: When ``vision_dynamic`` and ``text_dynamic`` disagree,
                or when a required detection source is missing.
        """
        if vision_dynamic is None:
            if visual is None:
                raise ValueError(
                    "_reconcile_dynamic_vision needs `vision_dynamic` or `visual`."
                )
            vision_dynamic = bool(getattr(visual, "_uses_dynamic_vision", False))
        if text_dynamic is None:
            if language_model is None:
                raise ValueError(
                    "_reconcile_dynamic_vision needs `text_dynamic` or `language_model`."
                )
            text_dynamic = bool(getattr(language_model, "_uses_rope_input", False))
        vision_dynamic = bool(vision_dynamic)
        text_dynamic = bool(text_dynamic)
        if vision_dynamic != text_dynamic:
            vision_path = getattr(config, "vision_mxq_path", "<unknown>")
            text_path = getattr(config, "text_mxq_path", "<unknown>")
            raise ValueError(
                "Qwen3-VL vision and text MXQs are a bundled release and cannot "
                "be swapped independently: visual._uses_dynamic_vision="
                f"{vision_dynamic} disagrees with language_model._uses_rope_input="
                f"{text_dynamic}. A dynamic-vision MXQ produces per-image RoPE "
                "tensors that the text MXQ must consume via its rope input; "
                "pairing a 3-input vision MXQ with a legacy 2-input text MXQ "
                "(or vice versa) silently corrupts image-boundary information. "
                f"vision_mxq_path={vision_path!r}, text_mxq_path={text_path!r}. "
                "Load a consistent Qwen3-VL release, or override both "
                "vision_mxq_path= and text_mxq_path= to a matching pair."
            )
        detected = vision_dynamic
        config_hint = bool(getattr(config, "dynamic_vision", False))
        if config_hint != detected:
            logger.warning_once(
                "Qwen3-VL config.dynamic_vision=%s disagrees with vision MXQ "
                "detection (%s); trusting the MXQ. Update the config or the "
                "shipped MXQ to match.",
                config_hint,
                detected,
            )
        config.dynamic_vision = detected
        return detected


class MobilintQwen3VLForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3VLPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3VLForConditionalGeneration,
):
    def __init__(self, config: MobilintQwen3VLConfig, *args, **kwargs):
        self._pretrained_only_base_init(config, *args, **kwargs)

        self.model = MobilintQwen3VLModel(config, _internal_call=True)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()

    def sync_dynamic_vision_from_model(self) -> bool:
        """Re-reconcile ``config.dynamic_vision`` from the loaded vision + text MXQs.

        Vision and text MXQs are a *bundled release* — one cannot be swapped
        independently, because a dynamic-vision MXQ produces per-image RoPE
        tensors that the text MXQ must consume via its rope input. This helper
        reads both compiled signatures and either promotes the agreed value
        onto ``self.config.dynamic_vision`` or raises when they disagree.

        The composite ``__init__`` already runs this reconciliation once, so
        calling this helper is only necessary when the model was loaded with a
        runtime override (e.g. ``vision_mxq_path=``) that could have swapped
        one side without the other. It re-checks the flags via
        :meth:`MobilintQwen3VLModel._reconcile_dynamic_vision`, so the same
        bundled-release invariant is enforced at both init time and post-load
        override time.

        Returns:
            The reconciled dynamic-vision flag now stored on ``self.config``.

        Raises:
            ValueError: When ``self.model.visual._uses_dynamic_vision`` and
                ``self.model.language_model._uses_rope_input`` disagree. The
                message names both flags, both MXQ paths, and tells the caller
                to load a consistent release or override both sides.
        """
        return MobilintQwen3VLModel._reconcile_dynamic_vision(
            self.config,
            visual=self.model.visual,
            language_model=self.model.language_model,
        )

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
