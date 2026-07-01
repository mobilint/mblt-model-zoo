import math
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
from ...utils.cache_utils import MobilintBeamCache, MobilintCache
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

# NPU audio encoder is compiled for a fixed 1-second window: 100 mel frames
# -> 13 audio tokens.
# NPU audio encoder 는 1초 (= 100 mel frame) 고정 입력으로 컴파일되어
# 13 audio token 을 출력한다.
_MEL_CHUNK_LEN = 100


def _qwen3_asr_tail_tokens(tail_frames: int) -> int:
    """Token count the upstream encoder would emit for ``tail_frames`` mel frames.

    Used to trim the zero-padded final chunk so audio embeddings stay aligned
    with the prompt's ``audio_token_id`` placeholders. Mirrors the three
    stride-2 conv reductions in upstream ``_get_feat_extract_output_lengths``.

    ``tail_frames`` 만큼 유효 mel frame 으로부터 upstream encoder 가 출력하는
    audio token 수. zero-padded 마지막 chunk 를 잘라 audio embedding 이
    프롬프트의 ``audio_token_id`` placeholder 와 1:1 정렬되도록 한다.
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

        # Zero-pad to a multiple of the fixed encoder window length.
        # encoder 고정 윈도우 길이의 배수가 되도록 zero-pad.
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

        # Trim padded tail tokens so audio embeddings match the prompt's
        # audio_token_id count.
        # zero-padded 꼬리 청크의 잉여 토큰 제거 - 프롬프트의 audio_token_id
        # 수와 audio embedding 을 1:1 정렬.
        if tail_frames > 0:
            valid_tail_tokens = _qwen3_asr_tail_tokens(tail_frames)
            emb_chunks[-1] = emb_chunks[-1][:, :valid_tail_tokens, :]

        audio_embeds_np = np.concatenate(emb_chunks, axis=1)
        audio_embeds = torch.tensor(
            audio_embeds_np,
            dtype=input_features.dtype,
            device=input_features.device,
        )

        # Upstream concatenates over dim=0, so the caller wants no batch dim.
        # upstream 이 dim=0 으로 concat 하므로 batch dim 없이 반환.
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

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args) -> MobilintBeamCache:
        """Return a beam-snapshot cache for Qwen3-ASR text generation."""
        del cache_implementation, max_cache_len, args
        configured_batch_size = max(1, int(batch_size), getattr(self.config, "max_batch_size", 1))
        if not hasattr(self, "_cache") or not isinstance(self._cache, MobilintBeamCache):
            self._cache = MobilintBeamCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        elif getattr(self._cache, "batch_size", 1) != configured_batch_size:
            self._cache = MobilintBeamCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        else:
            self._cache.reset()

        return self._cache

    def _resolve_source_indices(self, inputs_embeds: torch.Tensor) -> list[int]:
        """Return source ids for rows sharing the same audio-conditioned prompt embeddings."""
        source_indices: list[int] = []
        seen_signatures: list[torch.Tensor] = []
        for row in inputs_embeds.detach().cpu():
            source_index = next(
                (index for index, seen_row in enumerate(seen_signatures) if torch.equal(row, seen_row)),
                None,
            )
            if source_index is None:
                source_index = len(seen_signatures)
                seen_signatures.append(row.clone())
            source_indices.append(source_index)
        return source_indices

    def _beam_llm_forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        past_key_values: MobilintBeamCache,
        cache_position: torch.Tensor,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
    ) -> torch.Tensor:
        """Run Qwen3-ASR decoder by forwarding only suffixes missing from active cache."""
        if inputs_embeds.ndim != 3:
            raise ValueError(f"Expected rank-3 inputs_embeds for beam decode, got {tuple(inputs_embeds.shape)}")

        batch_size = int(inputs_embeds.shape[0])
        sequence_length = int(inputs_embeds.shape[1])
        if sequence_length <= 0:
            raise ValueError("Qwen3-ASR beam decode received an empty token sequence.")

        past_key_values.ensure_batch_size(batch_size)
        resolved_prefill_chunk_size = self.resolve_prefill_chunk_size(prefill_chunk_size)
        mxq_model = self.npu_backend.mxq_model
        logits_list: list[torch.Tensor] = []
        logits_by_target_tokens: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        self.npu_time = 0.0 if count_npu_time else None
        input_ids = input_ids.view(batch_size, -1)
        embedding_source_indices = self._resolve_source_indices(inputs_embeds)
        source_indices: list[int] = []
        for beam_index in range(batch_size):
            cached_source_index = past_key_values.get_beam_source_index(beam_index)
            source_indices.append(
                embedding_source_indices[beam_index] if cached_source_index is None else cached_source_index
            )
        past_key_values.set_beam_source_indices(source_indices)

        for beam_index in range(batch_size):
            previous_tokens = past_key_values.get_beam_tokens(beam_index)
            target_tokens = past_key_values.build_target_tokens(beam_index, input_ids[beam_index])
            source_index = source_indices[beam_index]
            target_key = (source_index, tuple(target_tokens))
            prefix_length = past_key_values.get_common_prefix_length(target_tokens, source_index=source_index)
            previous_length = len(previous_tokens)
            if prefix_length == len(target_tokens) and target_key in logits_by_target_tokens:
                logits_list.append(logits_by_target_tokens[target_key])
                past_key_values.commit_beam_tokens(beam_index, target_tokens)
                continue
            if prefix_length == len(target_tokens):
                prefix_length = max(0, len(target_tokens) - int(input_ids.shape[1]))
            local_start_index = max(0, prefix_length - previous_length)
            timing_phase = "prefill" if prefix_length == 0 else "decode"
            row_logits_ndarray: np.ndarray | None = None

            if prefix_length < previous_length:
                suffix_tokens = torch.tensor(
                    [target_tokens[prefix_length:]],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                row_embeds = self.embed_tokens(suffix_tokens)
            else:
                row_embeds = inputs_embeds[beam_index : beam_index + 1, local_start_index:, :]

            suffix_length = int(row_embeds.shape[1])
            if suffix_length <= 0:
                row_embeds = inputs_embeds[beam_index : beam_index + 1, -1:, :]
                suffix_length = 1
                prefix_length = max(0, len(target_tokens) - 1)

            row_embeds_numpy = row_embeds.detach().type(torch.float32).cpu().numpy()
            row_embeds_numpy = np.expand_dims(row_embeds_numpy, 1)
            num_suffix_chunks = math.ceil(suffix_length / resolved_prefill_chunk_size)

            for chunk_index in range(num_suffix_chunks):
                start_index = chunk_index * resolved_prefill_chunk_size
                end_index = min(start_index + resolved_prefill_chunk_size, suffix_length)
                cache_size = prefix_length + start_index
                chunk = row_embeds_numpy[:, :, start_index:end_index, :]

                if count_npu_time:
                    t0 = time.perf_counter()
                    result = mxq_model.infer([chunk], None, cache_size)
                    elapsed = time.perf_counter() - t0
                    assert self.npu_time is not None
                    self.npu_time += elapsed
                    self._record_npu_timing(timing_phase, elapsed)
                else:
                    result = mxq_model.infer([chunk], None, cache_size)

                assert result is not None, "mxq infer result is None!"
                row_logits_ndarray = result[0]

            past_key_values.commit_beam_tokens(beam_index, target_tokens)
            past_key_values.commit_active_tokens(target_tokens, source_index=source_index)
            if row_logits_ndarray is None:
                raise RuntimeError("Qwen3-ASR beam decode produced no logits.")
            row_logits = torch.tensor(row_logits_ndarray, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            while row_logits.ndim > 2:
                singleton_dims = [dim for dim, size in enumerate(row_logits.shape[:-1]) if int(size) == 1]
                if not singleton_dims:
                    raise ValueError(f"Unexpected Qwen3-ASR beam logits shape: {tuple(row_logits.shape)}")
                row_logits = row_logits.squeeze(singleton_dims[0])
            if row_logits.ndim == 2:
                row_logits = row_logits.unsqueeze(0)
            row_logits = row_logits[:, -int(input_ids.shape[1]) :, :]
            logits_list.append(row_logits)
            logits_by_target_tokens[target_key] = row_logits

        return torch.cat(logits_list, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids, inputs_embeds, or both.")

        explicit_no_cache = use_cache is False
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        if int(inputs_embeds.shape[0]) > 1 and not explicit_no_cache:
            use_cache = True

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", int(inputs_embeds.shape[0]), 0)

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

        if isinstance(past_key_values, MobilintBeamCache):
            if input_ids is None:
                raise ValueError("MobilintBeamCache requires input_ids to track Qwen3-ASR beam token histories.")
            logits = self._beam_llm_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                prefill_chunk_size=prefill_chunk_size,
                count_npu_time=count_npu_time,
            )
        else:
            logits = self.llm_forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                prefill_chunk_size=prefill_chunk_size,
                count_npu_time=count_npu_time,
                attention_mask=None,
                logits_to_keep=logits_to_keep,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
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
        # NPU TextModel already emits logits; route lm_head through Identity.
        # NPU TextModel 이 이미 logits 를 반환하므로 lm_head 는 통과 처리.
        self.lm_head = nn.Identity()

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.rope_deltas = None

    def get_cache_mxq_model(self):
        return self.model.get_mxq_model()

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args) -> MobilintBeamCache:
        """Return a beam-snapshot cache for the nested Qwen3-ASR thinker generation loop."""
        return self.model._get_cache(cache_implementation, batch_size, max_cache_len, *args)

    @wraps(Qwen3ASRThinkerForConditionalGeneration.forward)
    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        return_dict=None,
        count_npu_time: bool = False,
        **kwargs,
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            count_npu_time=count_npu_time,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if isinstance(logits, torch.Tensor) and logits.ndim == 4 and int(logits.shape[1]) == 1:
            logits = logits.squeeze(1)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)

        output = Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )
        if not return_dict:
            return output.to_tuple()

        return output


class MobilintQwen3ASRForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen3ASRPreTrainedModel,
    MobilintGenerationMixin,
    Qwen3ASRForConditionalGeneration,
):
    config_class = MobilintQwen3ASRConfig
    # The HF ASR pipeline keys inputs by `main_input_name`; default `input_ids`
    # would KeyError for an audio model.
    # HF ASR pipeline 이 `main_input_name` 으로 입력을 키잉하므로 audio
    # 모델은 default `input_ids` 대신 `input_features` 로 덮어쓴다.
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

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        pipeline_invocation = input_ids is None and input_features is not None
        if pipeline_invocation:
            if feature_attention_mask is None and attention_mask is not None:
                feature_attention_mask = attention_mask
                attention_mask = None
            input_ids, attention_mask = self._build_default_chat_input_ids(
                input_features, feature_attention_mask,
            )
        result = super().generate(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )
        if pipeline_invocation:
            if hasattr(result, "sequences"):
                sequences = self._strip_to_transcription(result.sequences)
                if not kwargs.get("return_dict_in_generate", False):
                    return sequences
                result.sequences = sequences
            elif torch.is_tensor(result):
                return self._strip_to_transcription(result)
        return result

    def _strip_to_transcription(self, sequences):
        """Drop the ``language <Lang><asr_text>`` preface from generated sequences.

        Lets the pipeline's tokenizer decode only the transcription. Rows
        without the marker pass through unchanged.

        생성 결과 앞의 ``language <Lang><asr_text>`` prefix 제거. pipeline 의
        decode 가 transcription 본문만 보도록 한다.
        """
        marker = self.config.thinker_config.asr_text_token_id
        trimmed = []
        for row in sequences:
            indices = (row == marker).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                start = int(indices[0].item()) + 1
                trimmed.append(row[start:])
            else:
                trimmed.append(row)
        max_len = max(t.shape[0] for t in trimmed)
        pad_id = self.config.thinker_config.text_config.eos_token_id or 0
        if isinstance(pad_id, list):
            pad_id = pad_id[0]
        padded = sequences.new_full((len(trimmed), max_len), pad_id)
        for i, t in enumerate(trimmed):
            padded[i, : t.shape[0]] = t
        return padded

    def _build_default_chat_input_ids(self, input_features, feature_attention_mask):
        """Fallback chat-template prompt when only audio features are provided.

        Used when the HF ASR pipeline calls ``generate`` without ``input_ids``.
        Mirrors what ``Qwen3ASRProcessor.apply_chat_template`` would produce.

        audio feature 만 받은 경우의 폴백 chat 프롬프트. HF ASR pipeline 이
        input_ids 없이 generate 를 호출할 때 사용. processor 가 만들
        프롬프트를 모델 내부에서 재현.
        """
        device = input_features.device
        if feature_attention_mask is not None:
            feat_lens = feature_attention_mask.sum(-1)
        else:
            feat_lens = torch.tensor([input_features.shape[-1]], device=device, dtype=torch.long)
        out_lens = (feat_lens - 1) // 2 + 1
        out_lens = (out_lens - 1) // 2 + 1
        out_lens = (out_lens - 1) // 2 + 1
        n_audio = int(out_lens.max().item())

        cfg = self.config.thinker_config
        # End the prompt at `<|im_start|>assistant\n` so the model continues
        # with `language <Lang><asr_text><transcription>` like upstream.
        # `<|im_start|>assistant\n` 으로 프롬프트를 종료 → 모델이 upstream 과
        # 같은 `language <Lang><asr_text><transcription>` 패턴으로 이어 생성.
        prompt = [
            cfg.im_start_token_id, cfg.user_token_id, cfg.newline_token_id,
            cfg.audio_start_token_id,
            *([cfg.audio_token_id] * n_audio),
            cfg.audio_end_token_id,
            cfg.newline_token_id, cfg.im_end_token_id, cfg.newline_token_id,
            cfg.im_start_token_id, cfg.assistant_token_id, cfg.newline_token_id,
        ]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask


AutoModel.register(MobilintQwen3ASRConfig, MobilintQwen3ASRForConditionalGeneration)
AutoModelForSpeechSeq2Seq.register(
    MobilintQwen3ASRConfig, MobilintQwen3ASRForConditionalGeneration
)


# Purpose: expose this model through `transformers.pipeline(
# "automatic-speech-recognition", ...)` on par with built-in ASR models
# (Whisper etc.) so users can swap to the Mobilint NPU build without rewriting
# their pipeline code. transformers 4.57.x does not ship a Qwen3-ASR entry in
# its seq2seq roster, so we add our class manually; otherwise the pipeline
# routes to a non-generative path and the model never runs.
# 목적: `transformers.pipeline("automatic-speech-recognition", ...)` 으로
# Whisper 같은 내장 ASR 모델과 동일한 방식으로 본 모델을 사용할 수 있게 한다.
# transformers 4.57.x 의 seq2seq 명단에 Qwen3-ASR 항목이 없어 직접 등록하지
# 않으면 pipeline 이 비-생성 경로로 빠져 모델이 실행되지 않는다.
try:
    from transformers.models.auto.modeling_auto import (  # noqa: E402
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    )

    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.setdefault(
        "mobilint-qwen3_asr", "MobilintQwen3ASRForConditionalGeneration"
    )
except ImportError:
    pass
