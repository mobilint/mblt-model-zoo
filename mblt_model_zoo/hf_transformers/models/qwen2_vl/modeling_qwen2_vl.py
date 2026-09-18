import inspect
from functools import lru_cache
from typing import Any, Union, cast

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForImageTextToText,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    VisionRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, can_return_tuple, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import (
    MobilintGenerationMixin,
    build_loss_kwargs_dynamic,
    mirror_output_fields,
    pop_loss_only_kwargs,
    upstream_positional_params,
    with_mobilint_generation_signature,
)
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen2_vl import (
    MobilintQwen2VLConfig,
    MobilintQwen2VLTextConfig,
    MobilintQwen2VLVisionConfig,
)

logger = logging.get_logger(__name__)


@lru_cache(maxsize=1)
def _upstream_qwen2_vl_vision_rotary_takes_position_ids() -> bool:
    """Return whether the installed Qwen2-VL vision RoPE takes position IDs."""
    try:
        signature = inspect.signature(VisionRotaryEmbedding.forward)
    except (TypeError, ValueError):
        return False
    parameters = list(signature.parameters.values())
    first = parameters[1].name if len(parameters) >= 2 else ""
    return first == "position_ids"


@lru_cache(maxsize=1)
def _upstream_qwen2_vl_uses_structured_vision_outputs() -> bool:
    """Check whether the installed Transformers expects ``visual()`` to return a model output.

    Returns:
        ``True`` when the installed upstream ``Qwen2VLModel.get_image_features`` reads
        ``vision_outputs.pooler_output``. ``False`` for older releases that expect ``visual()`` to
        return a raw tensor.
    """
    get_image_features = inspect.unwrap(Qwen2VLModel.get_image_features)
    code = getattr(get_image_features, "__code__", None)
    if code is not None:
        return "pooler_output" in code.co_names

    try:
        return "pooler_output" in inspect.getsource(get_image_features)
    except OSError:
        return True


class MobilintQwen2VLPreTrainedModel(PreTrainedModel):
    config: MobilintQwen2VLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")


class MobilintQwen2VisionTransformerPretrainedModel(MobilintModelMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLVisionConfig
    input_modalities = ("image", "video")

    def __init__(self, config: MobilintQwen2VLVisionConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        input_shapes = self.get_mxq_model().get_model_variant_handle(0).get_model_input_shape()
        self._uses_dynamic_vision = self._resolve_dynamic_vision_flag(len(input_shapes))
        if bool(getattr(config, "dynamic_vision", False)) != self._uses_dynamic_vision:
            logger.warning("Qwen2-VL vision MXQ signature overrides config.dynamic_vision=%s", config.dynamic_vision)
        config.dynamic_vision = self._uses_dynamic_vision
        if self._uses_dynamic_vision:
            head_dim = int(config.embed_dim) // int(config.num_heads)
            self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
            self._dynamic_input_slots = self._resolve_dynamic_input_slots(input_shapes, config)

    @staticmethod
    def _resolve_dynamic_vision_flag(num_mxq_inputs: int) -> bool:
        if num_mxq_inputs == 1:
            return False
        if num_mxq_inputs in (2, 3):
            return True
        raise ValueError(f"Qwen2-VL vision MXQ must expose 1, 2, or 3 inputs; got {num_mxq_inputs}.")

    @staticmethod
    def _resolve_dynamic_input_slots(input_shapes, config) -> dict[str, int]:
        widths = [int(shape[-1]) for shape in input_shapes]
        fold_width = int(config.in_channels) * int(config.temporal_patch_size) * int(config.patch_size) ** 2
        head_dim = int(config.embed_dim) // int(config.num_heads)
        expected = {"folded": fold_width}
        if len(input_shapes) == 2:
            expected["rope"] = 2 * (((head_dim + 63) // 64) * 64)
        else:
            expected["cos"] = head_dim
            expected["sin"] = head_dim
        slots = {}
        folded_matches = [idx for idx, actual in enumerate(widths) if actual == fold_width]
        if len(folded_matches) != 1:
            raise ValueError(f"Qwen2-VL dynamic vision widths {widths} do not uniquely identify folded={fold_width}.")
        slots["folded"] = folded_matches[0]
        for role, width in list(expected.items())[1:]:
            matches = [idx for idx, actual in enumerate(widths) if actual == width]
            if role == "rope" and len(matches) != 1:
                raise ValueError(f"Qwen2-VL dynamic vision widths {widths} do not match {role}={width}.")
            if role == "rope":
                slots[role] = matches[0]
        if len(input_shapes) == 3:
            remaining = [idx for idx in range(3) if idx != slots["folded"]]
            if any(widths[idx] != head_dim for idx in remaining):
                raise ValueError(f"Qwen2-VL dynamic vision widths {widths} do not match separate cos/sin width={head_dim}.")
            # The compiler forward signature is (images, cos, sin). Widths
            # cannot distinguish these two same-shaped tensors, so preserve
            # their defined positional order after locating folded pixels.
            slots["cos"], slots["sin"] = remaining
        return slots

    @property
    def dtype(self) -> torch.dtype:
        """Expose the MXQ vision input dtype expected by upstream Qwen2-VL helpers."""
        return torch.float32

    @property
    def spatial_merge_size(self) -> int:
        """Expose the merge factor expected by upstream Qwen2-VL helpers."""
        return int(self.config.spatial_merge_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        """Run the compiled vision encoder and adapt to HF's vision output contract.

        Args:
            hidden_states: Processor-produced image or video patch tensor.
            grid_thw: Temporal, height, and width grid metadata for each image or video.
            **kwargs: Additional Transformers kwargs. ``return_dict`` falls back to
                ``self.config.return_dict`` when the installed upstream expects structured outputs.

        Returns:
            By default, a value matching the installed upstream Qwen2-VL contract:
            ``BaseModelOutputWithPooling`` on newer Transformers and a raw tensor on older ones.
            The Mobilint backend does not expose per-patch hidden states or attentions, so those
            structured fields are returned as ``None`` when present.
        """
        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None and _upstream_qwen2_vl_uses_structured_vision_outputs():
            return_dict = self.config.return_dict
        del kwargs
        merged_hidden_states = self._encode_images(hidden_states, grid_thw)
        structured_outputs = BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=merged_hidden_states,
            hidden_states=None,
            attentions=None,
        )
        if return_dict is True:
            return structured_outputs
        if _upstream_qwen2_vl_uses_structured_vision_outputs():
            if return_dict is False:
                return structured_outputs.to_tuple()
            return structured_outputs
        return merged_hidden_states

    def _preprocess_image_tokens(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Convert one Qwen2-VL image patch sequence to the MXQ vision input layout."""
        gt, gh, gw = grid_thw.tolist()

        c = 3
        pt = 2
        mh = 2
        mw = 2
        gh2 = gh // 2
        gw2 = gw // 2
        ph = pw = int((hidden_states.shape[-1] // (pt * c)) ** 0.5)

        expected_tokens = gt * gh2 * gw2 * mh * mw
        expected_hidden = c * pt * ph * pw
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                f"Unexpected pixel token count for Qwen2-VL vision input: {hidden_states.shape[0]} vs {expected_tokens}"
            )
        if hidden_states.shape[1] != expected_hidden:
            raise ValueError(
                f"Unexpected pixel hidden size for Qwen2-VL vision input: {hidden_states.shape[1]} vs {expected_hidden}"
            )

        hidden_states = hidden_states.view(gt, gh2, gw2, mh, mw, c, pt, ph, pw)
        hidden_states = hidden_states.permute(0, 1, 2, 7, 3, 4, 8, 6, 5).contiguous()
        return hidden_states.view(gt, gh2 * gw2 * ph, mh * mw * pw, pt * c).squeeze(0)

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
            raise ValueError(f"Unexpected total Qwen2-VL pixel token count: {hidden_states.shape[0]} vs {offset}")
        return chunks

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            shape = (h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size)
            hpos = hpos.reshape(shape).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(shape).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack((hpos, wpos), dim=-1).repeat(int(t), 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max())
        self._materialize_vision_rotary()
        if _upstream_qwen2_vl_vision_rotary_takes_position_ids():
            inv_freq = self.rotary_pos_emb.inv_freq
            device = torch.device("cpu") if inv_freq.device.type == "meta" else inv_freq.device
            position_ids = torch.arange(max_grid_size, device=device, dtype=inv_freq.dtype)
            freq_table = self.rotary_pos_emb(position_ids)
        else:
            freq_table = self.rotary_pos_emb(max_grid_size)
        return freq_table[pos_ids.to(freq_table.device)].flatten(1)

    def _materialize_vision_rotary(self) -> None:
        """Rebuild upstream's runtime-only vision frequencies off the meta device."""
        dim = (int(self.config.embed_dim) // int(self.config.num_heads)) // 2
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
        )
        self.rotary_pos_emb.inv_freq = inv_freq

    def _build_vision_rotate_tensor(self, grid_thw: torch.Tensor) -> np.ndarray:
        rotary = self._rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos_val, sin_val = emb.cos(), emb.sin()
        dim = int(emb.shape[-1])
        half = dim // 2
        tgt_half = ((dim + 63) // 64) * 64
        packed = torch.zeros((emb.shape[0], 2 * tgt_half), dtype=torch.float32, device=emb.device)
        packed[:, 0:dim:2] = cos_val[:, :half]
        packed[:, 1:dim:2] = -sin_val[:, :half]
        packed[:, tgt_half : tgt_half + dim : 2] = sin_val[:, half:]
        packed[:, tgt_half + 1 : tgt_half + dim : 2] = cos_val[:, half:]
        return packed.reshape(1, -1, 2 * tgt_half).cpu().numpy()

    def _prepare_dynamic_npu_inputs(self, hidden_states: torch.Tensor, grid: torch.Tensor) -> list[np.ndarray]:
        grid = grid.unsqueeze(0) if grid.ndim == 1 else grid
        folded = hidden_states.transpose(0, 1).reshape(1, hidden_states.shape[1], 1, hidden_states.shape[0])
        folded_np = folded.squeeze(2).permute(0, 2, 1).float().cpu().numpy()
        payloads: list[np.ndarray | None] = [None] * len(self._dynamic_input_slots)
        payloads[self._dynamic_input_slots["folded"]] = folded_np
        rotary = self._rot_pos_emb(grid)
        emb = torch.cat((rotary, rotary), dim=-1)
        if "rope" in self._dynamic_input_slots:
            payloads[self._dynamic_input_slots["rope"]] = self._build_vision_rotate_tensor(grid)
        else:
            payloads[self._dynamic_input_slots["cos"]] = emb.cos().reshape(1, -1, emb.shape[-1]).float().cpu().numpy()
            payloads[self._dynamic_input_slots["sin"]] = emb.sin().reshape(1, -1, emb.shape[-1]).float().cpu().numpy()
        return cast(list[np.ndarray], payloads)

    def _flatten_encoder_output(
        self,
        output: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Normalize Qwen2-VL MXQ vision output to `(total_image_tokens, hidden_size)`."""
        if output.ndim >= 3 and int(output.shape[0]) == batch_size:
            return output.reshape(-1, int(output.shape[-1]))
        if output.ndim >= 3 and int(output.shape[0]) == 1:
            return output.squeeze(0).reshape(-1, int(output.shape[-1]))
        if output.ndim == 2:
            return output
        raise ValueError(f"Unexpected Qwen2-VL vision output shape: {tuple(output.shape)}")

    def _encode_images(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Run Qwen2-VL vision encoding with core-mode-specific batch handling."""
        uses_dynamic = bool(getattr(self, "_uses_dynamic_vision", False))
        chunks = self._split_hidden_states_by_grid(hidden_states, grid_thw)
        mxq_inputs = (
            [self._prepare_dynamic_npu_inputs(chunk, grid) for chunk, grid in zip(chunks, grid_thw)]
            if uses_dynamic
            else [self._preprocess_image_tokens(chunk, grid) for chunk, grid in zip(chunks, grid_thw)]
        )
        npu_backend = getattr(self, "npu_backend", None)
        core_mode = getattr(npu_backend, "core_mode", getattr(self.config, "core_mode", "auto"))
        if core_mode == "multi" and len(mxq_inputs) > 1:
            if uses_dynamic:
                raise NotImplementedError("Batched dynamic Qwen2-VL vision MXQ inputs are not supported")
            batched_inputs = torch.stack(mxq_inputs, dim=0)
            return self._flatten_encoder_output(self.mxq_forward(batched_inputs), batch_size=len(mxq_inputs))

        if uses_dynamic:
            outputs = []
            for mxq_input in mxq_inputs:
                result = self.npu_backend.mxq_models[0].infer(mxq_input)
                if result is None:
                    raise RuntimeError("Qwen2-VL dynamic vision MXQ inference returned None")
                outputs.append(self._flatten_encoder_output(torch.as_tensor(result[0]), batch_size=1))
        else:
            outputs = [self._flatten_encoder_output(self.mxq_forward(mxq_input), batch_size=1) for mxq_input in mxq_inputs]
        return torch.cat(outputs, dim=0)


class MobilintQwen2VLRotaryEmbedding:
    """Runtime Qwen2-VL MRoPE using upstream's chunked mrope_section layout."""

    def __init__(self, config: MobilintQwen2VLTextConfig):
        self.head_dim = int(config.hidden_size) // int(config.num_attention_heads)
        scaling = getattr(config, "rope_scaling", None) or {}
        self.rope_theta = float(getattr(config, "rope_theta", None) or scaling.get("rope_theta", 10000.0))
        section = scaling.get("mrope_section")
        if section is not None and sum(section) * 2 != self.head_dim:
            raise ValueError(f"Qwen2-VL mrope_section={section} does not cover head_dim={self.head_dim}")
        self.mrope_section = section
        self.inv_freq = None
        self.pe_size = 2 * (((self.head_dim + 63) // 64) * 64)

    def _get_inv_freq(self) -> torch.Tensor:
        if self.inv_freq is None or self.inv_freq.device.type == "meta":
            self.inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device="cpu") / self.head_dim)
            )
        return self.inv_freq

    def __call__(self, position_ids: torch.Tensor) -> np.ndarray:
        inv_freq = self._get_inv_freq()
        position_ids = position_ids[:3].to(dtype=torch.float32, device=inv_freq.device)
        freqs = torch.einsum("d,nbs->nbsd", inv_freq, position_ids)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        if self.mrope_section is not None:
            sections = [value * 2 for value in self.mrope_section]
            cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
            sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
        else:
            cos, sin = cos[0], sin[0]
        dim = int(cos.shape[-1])
        half = dim // 2
        tgt_half = ((dim + 63) // 64) * 64
        packed = torch.zeros((*cos.shape[:-1], 2 * tgt_half), dtype=torch.float32)
        packed[..., 0:dim:2] = cos[..., :half]
        packed[..., 1:dim:2] = -sin[..., :half]
        packed[..., tgt_half : tgt_half + dim : 2] = sin[..., half:]
        packed[..., tgt_half + 1 : tgt_half + dim : 2] = cos[..., half:]
        return packed.cpu().numpy()


class MobilintQwen2VLTextModel(MobilintModelMixin, MobilintGenerationMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLTextConfig
    input_modalities = ("text",)

    def __init__(self, config: MobilintQwen2VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        input_shapes = self.get_mxq_model().get_model_variant_handle(0).get_model_input_shape()
        self._uses_rope_input = len(input_shapes) == 2
        if len(input_shapes) not in (1, 2):
            raise ValueError(f"Qwen2-VL text MXQ must expose 1 static or 2 dynamic inputs; got {len(input_shapes)}")
        self._mobilint_rotary_emb = MobilintQwen2VLRotaryEmbedding(config) if self._uses_rope_input else None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.LongTensor, None] = None,
        past_key_values: Union[MobilintCache, None] = None,
        inputs_embeds: Union[torch.FloatTensor, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        npu_prefill_chunk_size: Union[int, None] = None,
        count_npu_time: bool = False,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        # Route attention_mask through so batch>1 hits MobilintModelMixin._llm_forward_batch.
        # Qwen2-VL text decoder has the same structural contract as plain Qwen2 here.
        effective_attention_mask = self.resolve_batched_attention_mask(inputs_embeds, attention_mask)
        if self._uses_rope_input and effective_attention_mask is not None:
            raise NotImplementedError("Batched Qwen2-VL text MXQs with an external RoPE input are not supported")

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor,
                torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device),
            )

        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")

        position_embeddings = None
        if self._uses_rope_input:
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
            assert self._mobilint_rotary_emb is not None
            position_embeddings = self._mobilint_rotary_emb(position_ids)

        logits = self.llm_forward(
            inputs_embeds,
            past_key_values,
            cache_position,
            npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            attention_mask=effective_attention_mask,
            logits_to_keep=logits_to_keep,
            extra_inputs=position_embeddings,
        )

        if not return_dict:
            return tuple(v for v in [logits, past_key_values, None, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class MobilintQwen2VLModel(PretrainedOnlyMixin, MobilintQwen2VLPreTrainedModel, Qwen2VLModel):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        MobilintQwen2VLPreTrainedModel.__init__(self, config, *args, **kwargs)
        self.visual = MobilintQwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, _internal_call=True
        )
        self.language_model = MobilintQwen2VLTextModel._from_config(
            config.text_config,
            _internal_call=True,
        )
        self._reconcile_dynamic_vision(config)
        self.rope_deltas = None  # cache rope_deltas here

    def _reconcile_dynamic_vision(self, config: MobilintQwen2VLConfig) -> bool:
        vision_dynamic = bool(getattr(self.visual, "_uses_dynamic_vision", False))
        text_dynamic = bool(getattr(self.language_model, "_uses_rope_input", False))
        if vision_dynamic != text_dynamic:
            raise ValueError(
                "Qwen2-VL vision and text MXQs must agree on dynamic mode: "
                f"vision_dynamic={vision_dynamic}, text_dynamic={text_dynamic}. "
                "Load a matching vision/text MXQ pair; a dynamic vision MXQ "
                "requires a text MXQ with an external RoPE input."
            )
        config.dynamic_vision = vision_dynamic
        return vision_dynamic

        # Initialize weights and apply final processing
        self.post_init()


class MobilintQwen2VLForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen2VLPreTrainedModel,
    MobilintGenerationMixin,
    Qwen2VLForConditionalGeneration,
):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        self._pretrained_only_base_init(config, *args, **kwargs)

        self.model = MobilintQwen2VLModel(config, _internal_call=True)
        self.config.dynamic_vision = bool(self.model._reconcile_dynamic_vision(config))
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()

    def get_cache_mxq_model(self):
        return self.model.language_model.get_mxq_model()

    def sync_dynamic_vision_from_model(self) -> bool:
        """Publish the detected vision-MXQ mode for an overridden processor."""
        return self.model._reconcile_dynamic_vision(self.config)

    @with_mobilint_generation_signature(
        Qwen2VLForConditionalGeneration.prepare_inputs_for_generation,
        "count_npu_time",
        "npu_prefill_chunk_size",
    )
    def prepare_inputs_for_generation(
        self,
        *args: object,
        count_npu_time: bool = False,
        npu_prefill_chunk_size: int | None = None,
        **kwargs: object,
    ):
        """Prepare generation inputs while preserving Mobilint timing kwargs.

        Args:
            *args: Positional arguments forwarded to the upstream Qwen2-VL generation helper.
            count_npu_time: Whether Mobilint decoder NPU time should be accumulated.
            npu_prefill_chunk_size: Optional prefill chunk size forwarded to Mobilint generation.
            **kwargs: Keyword arguments forwarded to the upstream Qwen2-VL generation helper.

        Returns:
            Model inputs for a generation step.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["count_npu_time"] = count_npu_time
        if npu_prefill_chunk_size is not None:
            model_inputs["npu_prefill_chunk_size"] = npu_prefill_chunk_size
        return model_inputs

    @with_mobilint_generation_signature(Qwen2VLForConditionalGeneration.forward, "count_npu_time")
    @can_return_tuple
    def forward(
        self,
        *args: Any,
        count_npu_time: bool = False,
        **kwargs: Any,
    ) -> Union[tuple, Qwen2VLCausalLMOutputWithPast]:
        """Route ``logits_to_keep`` to the Mobilint text decoder.

        Upstream ``Qwen2VLForConditionalGeneration.forward`` extracts ``logits_to_keep``
        as a named argument and performs its own final slice on the text model output,
        which bypasses the Mobilint decoder's position selection. To keep that decoder
        in charge of picking positions, we pop ``logits_to_keep`` here and thread it
        into ``self.model`` via kwargs (upstream ``Qwen2VLModel.forward`` forwards its
        own ``**kwargs`` to the text model). All other arguments follow the upstream
        signature by way of ``@with_mobilint_generation_signature``, so upstream
        additions such as ``position_ids`` continue to pass through unchanged.

        Tuple mode: ``@can_return_tuple`` strips ``return_dict`` from kwargs before
        the wrapper body runs (so ``self.model`` never returns a tuple) and converts
        the assembled ``Qwen2VLCausalLMOutputWithPast`` back to a tuple when
        ``return_dict=False`` was requested — matching the upstream forward's
        contract.

        Dynamic adaptation:
            * Loss kwargs are built via :func:`build_loss_kwargs_dynamic`, so
              upstream additions like ``num_items_in_batch`` / ``shift_labels``
              flow through when the loss function accepts them.
            * The returned ``Qwen2VLCausalLMOutputWithPast`` is assembled by
              :func:`mirror_output_fields`, so new output fields (e.g. a future
              ``image_hidden_states``) are mirrored from the upstream model
              output automatically instead of requiring wrapper edits.

        Performance: the default ``logits_to_keep=0`` (keep-all) matches HF but on
        last-only MXQ triggers a size-1 infer per input token. ``.generate()`` is
        safe (HF passes ``logits_to_keep=1``); manual ``.forward()`` callers doing
        perplexity eval / logit collection inherit this cost on last-only builds.
        """
        positional_params = upstream_positional_params(Qwen2VLForConditionalGeneration.forward)
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
        # stripped BEFORE ``self.model`` is called: upstream ``Qwen2VLModel``
        # forwards its own ``**kwargs`` to ``self.language_model``, and
        # ``MobilintQwen2VLTextModel.forward`` declares no ``**kwargs`` sink,
        # so leaking these into that call would raise ``TypeError``.
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
            Qwen2VLCausalLMOutputWithPast,
            outputs,
            loss=loss,
            logits=logits,
        )


AutoModel.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
