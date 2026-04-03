from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen3_vl import (
    MobilintQwen3VLConfig,
    MobilintQwen3VLTextConfig,
    MobilintQwen3VLVisionConfig,
)

logger = logging.get_logger(__name__)


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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the NPU vision encoder.

        The compiled encoder expects a fixed-shape tensor with the following flow:

        1. HF processor output: `(256, 1536)` for a 224x224 image
        2. Runtime repreprocess: `(1, 6, 1024, 64)`
        3. Final MXQ input: `(1024, 64, 6)`
        """
        del kwargs
        if hidden_states.ndim < 2:
            raise ValueError(f"Expected pixel tensor with rank >=2, got shape {tuple(hidden_states.shape)}")

        npu_inputs = self._prepare_npu_inputs(hidden_states, grid_thw)
        encoder_outputs = self.get_mxq_model().infer(npu_inputs)
        if encoder_outputs is None:
            raise RuntimeError("Vision MXQ inference returned None.")

        image_embeds, deepstack_embeds = self._reorder_encoder_outputs(encoder_outputs, hidden_states.device)
        return image_embeds, deepstack_embeds

    def _repreprocess_pixel_values(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Match the runtime `repreprocess_pixel_values` layout for Qwen3-VL vision input."""
        gt, gh, gw = grid_thw[0].tolist()
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
                "Unexpected pixel token count for Qwen3-VL vision input: "
                f"{hidden_states.shape[0]} vs {expected_tokens}"
            )
        if hidden_states.shape[1] != expected_hidden:
            raise ValueError(
                "Unexpected pixel hidden size for Qwen3-VL vision input: "
                f"{hidden_states.shape[1]} vs {expected_hidden}"
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
        return processed.cpu().numpy().astype(np.float32)

    def _reorder_encoder_outputs(
        self,
        encoder_outputs: list[np.ndarray],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(encoder_outputs) < 4:
            raise ValueError(f"Expected at least 4 encoder outputs, got {len(encoder_outputs)}")

        image_embeds = torch.tensor(
            np.squeeze(encoder_outputs[0]),
            dtype=torch.float32,
            device=device,
        )
        deepstack_embeds = [
            torch.tensor(np.squeeze(encoder_outputs[2]), dtype=torch.float32, device=device),
            torch.tensor(np.squeeze(encoder_outputs[3]), dtype=torch.float32, device=device),
            torch.tensor(np.squeeze(encoder_outputs[1]), dtype=torch.float32, device=device),
        ]
        return image_embeds, deepstack_embeds


class MobilintQwen3VLTextModel(MobilintModelMixin, MobilintGenerationMixin, MobilintQwen3VLPreTrainedModel):
    config: MobilintQwen3VLTextConfig
    input_modalities = ("text",)

    def __init__(self, config: MobilintQwen3VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.num_deepstack_layers = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(torch.LongTensor, torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ))

        logits = self.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            count_npu_time=count_npu_time,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
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
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
    ) -> torch.Tensor:
        """Run the dual-input MXQ decoder.

        Args:
            inputs_embeds: Token embeddings of shape `(batch, seq_len, hidden)`.
            deepstack_visual_embeds: Optional deepstack features by layer.
            visual_pos_masks: Optional visual token mask.
            past_key_values: Mobilint KV cache.
            cache_position: Cache position range.
            prefill_chunk_size: Optional chunk size.
            count_npu_time: Whether to accumulate NPU time.

        Returns:
            Decoder logits for the most recent token positions.
        """
        if inputs_embeds.ndim != 3:
            raise ValueError(f"Expected inputs_embeds rank 3, got shape {tuple(inputs_embeds.shape)}")
        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Mobilint Qwen3-VL currently supports batch size 1 only.")

        deepstack_tensor = self._build_deepstack_tensor(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        inputs_np = inputs_embeds.type(torch.float32).cpu().numpy()
        deepstack_np = deepstack_tensor.type(torch.float32).cpu().numpy()

        resolved_prefill_chunk_size = self.resolve_prefill_chunk_size(prefill_chunk_size)
        num_chunks = int(np.ceil(inputs_np.shape[1] / resolved_prefill_chunk_size))

        mxq_model = self.get_mxq_model()
        self.npu_time = 0.0 if count_npu_time else None
        logits_ndarray = None

        for chunk_idx in range(num_chunks):
            start_index = chunk_idx * resolved_prefill_chunk_size
            end_index = min(start_index + resolved_prefill_chunk_size, inputs_np.shape[1])
            cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()

            inputs_chunk = inputs_np[:, start_index:end_index, :]
            deepstack_chunk = deepstack_np[:, start_index:end_index, :]

            if count_npu_time:
                import time

                t1 = time.perf_counter()
                result = mxq_model.infer([inputs_chunk, deepstack_chunk], None, cache_size)
                self.npu_time += time.perf_counter() - t1
            else:
                result = mxq_model.infer([inputs_chunk, deepstack_chunk], None, cache_size)

            if result is None:
                raise RuntimeError("Text MXQ inference returned None.")
            logits_ndarray = result[0]

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        if logits_ndarray is None:
            raise RuntimeError("Text MXQ inference did not produce logits.")

        return torch.tensor(logits_ndarray, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

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

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        """Encode images and return split image embeddings plus deepstack features."""
        if image_grid_thw is None:
            raise ValueError("image_grid_thw must not be None.")

        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.config.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        """Encode videos using the same MXQ vision path as images."""
        return self.get_image_features(pixel_values_videos, video_grid_thw)


class MobilintQwen3VLForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3VLPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3VLForConditionalGeneration,
):
    def __init__(self, config: MobilintQwen3VLConfig, *args, **kwargs):
        """Initialize the multimodal model and bypass HF lm_head."""
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)
        
        self.model = MobilintQwen3VLModel(config, _internal_call=True)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()

    def get_cache_mxq_model(self):
        return self.model.language_model.get_mxq_model()

    def forward(
        self,
        *args,
        count_npu_time: bool = False,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        kwargs["count_npu_time"] = count_npu_time
        return super().forward(*args, **kwargs)


AutoModel.register(MobilintQwen3VLConfig, MobilintQwen3VLForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintQwen3VLConfig, MobilintQwen3VLForConditionalGeneration)
