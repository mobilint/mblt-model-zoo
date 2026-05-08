import time
from functools import wraps
from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForSpeechSeq2Seq,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from ._errors import guard_qwen_asr_import

with guard_qwen_asr_import():
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRPreTrainedModel,
        Qwen3ASRThinkerCausalLMOutputWithPast,
        Qwen3ASRThinkerForConditionalGeneration,
    )
from .configuration_qwen3_asr import (
    MobilintQwen3ASRAudioEncoderConfig,
    MobilintQwen3ASRConfig,
    MobilintQwen3ASRTextConfig,
    MobilintQwen3ASRThinkerConfig,
)

logger = logging.get_logger(__name__)

# 1 second of mel = 100 frames -> 13 audio tokens; NPU encoder fixed-input length.
_MEL_CHUNK_LEN = 100


def _qwen3_asr_tail_tokens(tail_frames: int) -> int:
    """Audio token count upstream Qwen3-ASR emits for a partial tail chunk.

    Mirrors the tail term of
    ``qwen_asr.core.transformers_backend.modeling_qwen3_asr._get_feat_extract_output_lengths``:
    three stride-2 conv stages reduce ``tail_frames`` mel frames down to a
    smaller token count. The NPU encoder always returns 13 tokens because its
    input is zero-padded to ``_MEL_CHUNK_LEN``, so the last chunk's output must
    be sliced to this value to keep audio embeddings aligned with the
    ``audio_token_id`` placeholders the processor pre-allocated in the prompt.

    Args:
        tail_frames: Valid mel-frame count in the partial last chunk, in
            ``[0, _MEL_CHUNK_LEN)``. ``0`` means no partial tail (the input
            length is an exact multiple of ``_MEL_CHUNK_LEN``).

    Returns:
        Number of audio tokens that correspond to ``tail_frames`` valid mel
        frames. Returns ``0`` for ``tail_frames <= 0``.
    """
    if tail_frames <= 0:
        return 0
    n = (tail_frames - 1) // 2 + 1
    n = (n - 1) // 2 + 1
    n = (n - 1) // 2 + 1
    return n


class MobilintQwen3ASRPreTrainedModel(Qwen3ASRPreTrainedModel):
    config: MobilintQwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}


class MobilintQwen3ASRAudioEncoder(MobilintModelMixin, MobilintQwen3ASRPreTrainedModel):
    config: MobilintQwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = ("audio",)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
        if input_features.ndim != 3 or input_features.shape[0] != 1:
            raise NotImplementedError(
                f"Mobilint Qwen3-ASR audio encoder supports batch=1 only, "
                f"got shape {tuple(input_features.shape)}"
            )

        T_total = input_features.shape[-1]
        T_valid = int(feature_lens[0].item()) if feature_lens is not None else T_total

        # Zero-pad up to the next multiple of _MEL_CHUNK_LEN.
        n_full      = T_valid // _MEL_CHUNK_LEN
        tail_frames = T_valid % _MEL_CHUNK_LEN
        n_chunks    = n_full + (1 if tail_frames > 0 else 0)
        T_padded    = n_chunks * _MEL_CHUNK_LEN

        x = input_features[..., :T_valid]
        if T_padded > T_valid:
            x = F.pad(x, (0, T_padded - T_valid))

        # NCHW: (1, 1, 128, T_padded). NPU expects (1, 1, 128, 100) per chunk.
        x = x.unsqueeze(1)

        mxq = self.get_mxq_model()
        emb_chunks: list[np.ndarray] = []
        for i in range(n_chunks):
            chunk = x[:, :, :, i * _MEL_CHUNK_LEN : (i + 1) * _MEL_CHUNK_LEN]
            chunk_np = chunk.type(torch.float32).cpu().numpy()
            result = mxq.infer([chunk_np], None, 0)
            if result is None:
                raise RuntimeError("Audio MXQ inference returned None.")
            # (1, 1024, 1, 13) NHWC -> squeeze axis=2 -> (1, 1024, 13) -> (1, 13, 1024).
            emb_np = np.transpose(np.squeeze(result[0], axis=2), (0, 2, 1))
            emb_chunks.append(emb_np)

        # The last chunk was padded to _MEL_CHUNK_LEN, so the encoder still
        # emits 13 tokens for it. Slice to the count upstream Qwen3-ASR derives
        # from the real tail length so audio embeddings stay 1:1 with the
        # processor's audio_token_id placeholders.
        if tail_frames > 0:
            valid_tail_tokens = _qwen3_asr_tail_tokens(tail_frames)
            emb_chunks[-1] = emb_chunks[-1][:, :valid_tail_tokens, :]

        audio_embeds_np = np.concatenate(emb_chunks, axis=1)
        audio_embeds = torch.tensor(
            audio_embeds_np,
            dtype=input_features.dtype,
            device=input_features.device,
        )

        # Upstream concatenates over dim=0; return without batch dim.
        return BaseModelOutput(last_hidden_state=audio_embeds.squeeze(0))


class MobilintQwen3ASRTextModel(
    MobilintModelMixin,
    MobilintGenerationMixin,
    MobilintQwen3ASRPreTrainedModel,
):
    config: MobilintQwen3ASRTextConfig
    input_modalities = ("text",)

    def __init__(self, config: MobilintQwen3ASRTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = cast(
                torch.LongTensor,
                torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                ),
            )

        logits = self._llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            count_npu_time=count_npu_time,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
        )

    def _llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds.ndim != 3:
            raise ValueError(
                f"Expected inputs_embeds rank 3, got shape {tuple(inputs_embeds.shape)}"
            )
        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError(
                "Mobilint Qwen3-ASR text model supports batch=1 only."
            )

        inputs_np = inputs_embeds.type(torch.float32).cpu().numpy()
        resolved_chunk = self.resolve_prefill_chunk_size(prefill_chunk_size)
        num_chunks = int(np.ceil(inputs_np.shape[1] / resolved_chunk))

        mxq = self.get_mxq_model()
        self.npu_time = 0.0 if count_npu_time else None
        logits_ndarray = None

        for chunk_idx in range(num_chunks):
            start = chunk_idx * resolved_chunk
            end   = min(start + resolved_chunk, inputs_np.shape[1])
            cache_size = (
                0 if past_key_values is None else past_key_values.get_seq_length()
            )

            chunk = inputs_np[:, start:end, :]

            if count_npu_time:
                t0 = time.perf_counter()
                result = mxq.infer([chunk], None, cache_size)
                self.npu_time += time.perf_counter() - t0
            else:
                result = mxq.infer([chunk], None, cache_size)

            if result is None:
                raise RuntimeError("Text MXQ inference returned None.")
            logits_ndarray = result[0]

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start:end])

        if logits_ndarray is None:
            raise RuntimeError("Text MXQ inference did not produce logits.")

        return torch.tensor(
            logits_ndarray,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )


class MobilintQwen3ASRThinkerForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3ASRPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3ASRThinkerForConditionalGeneration,
):
    config: MobilintQwen3ASRThinkerConfig

    def __init__(self, config: MobilintQwen3ASRThinkerConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)

        self.audio_tower = MobilintQwen3ASRAudioEncoder._from_config(
            config.audio_config, _internal_call=True
        )
        self.model = MobilintQwen3ASRTextModel._from_config(
            config.text_config, _internal_call=True
        )
        # NPU TextModel returns logits directly; lm_head is a pass-through.
        self.lm_head = nn.Identity()

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.rope_deltas = None

    def get_cache_mxq_model(self):
        return self.model.get_mxq_model()

    @wraps(Qwen3ASRThinkerForConditionalGeneration.forward)
    def forward(
        self,
        *args: object,
        count_npu_time: bool = False,
        **kwargs: object,
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        kwargs["count_npu_time"] = count_npu_time
        return super().forward(*args, **kwargs)


class MobilintQwen3ASRForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3ASRPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3ASRForConditionalGeneration,
):
    config_class = MobilintQwen3ASRConfig
    # HF ASR pipeline pops by `model.main_input_name`; default is `input_ids`,
    # which would raise KeyError for an audio model.
    main_input_name = "input_features"

    def __init__(self, config: MobilintQwen3ASRConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)

        self.config = config
        self.thinker = MobilintQwen3ASRThinkerForConditionalGeneration._from_config(
            config.thinker_config, _internal_call=True
        )
        self.post_init()

    def get_cache_mxq_model(self):
        return self.thinker.get_cache_mxq_model()


AutoModel.register(MobilintQwen3ASRConfig, MobilintQwen3ASRForConditionalGeneration)
AutoModelForSpeechSeq2Seq.register(
    MobilintQwen3ASRConfig, MobilintQwen3ASRForConditionalGeneration
)
