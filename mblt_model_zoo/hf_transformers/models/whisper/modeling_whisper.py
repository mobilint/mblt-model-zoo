import os
from typing import Any, Union, cast

import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSpeechSeq2Seq
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.models.whisper.modeling_whisper import (
    WhisperPositionalEmbedding,
    _compute_mask_indices,
    shift_tokens_right,
)
from transformers.utils.generic import logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import (
    MobilintWhisperCache,
    append_whisper_beam_debug_event,
    is_whisper_beam_debug_trace_enabled,
)
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_whisper import MobilintWhisperConfig

logger = logging.get_logger(__name__)

class MobilintWhisperPreTrainedModel(PreTrainedModel):
    config: MobilintWhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = ("audio", "text")

class MobilintWhisperEncoder(MobilintModelMixin, MobilintWhisperPreTrainedModel):
    npu_backend_prefix = "encoder_"
    
    def __init__(self, config: MobilintWhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
       
        # Make anonymous functions just for adding attributes
        self.conv1 = lambda: None
        self.conv2 = lambda: None
        
        # Used in self.forward and WhisperGenerationMixin
        self.conv1.stride = [1]
        self.conv2.stride = [2]
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                "Whisper expects the mel input features to be of length "
                f"{expected_seq_length}, but found {input_features.shape[-1]}. "
                f"Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")
        
        hidden_states = self.mxq_forward(input_features.permute(0, 2, 1))
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif hidden_states.ndim == 4 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        
        if not return_dict:
            return tuple(v for v in [hidden_states,] if v is not None)
        return BaseModelOutput(
            last_hidden_state=cast(torch.FloatTensor, hidden_states), hidden_states=None, attentions=None
        )

class MobilintWhisperDecoder(MobilintModelMixin, MobilintWhisperPreTrainedModel):
    npu_backend_prefix = "decoder_"
    main_input_name = "input_ids"
    
    def __init__(self, config: MobilintWhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.embed_positions = WhisperPositionalEmbedding(config.max_target_positions, config.d_model)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def _embed_token_suffix(
        self,
        token_history: list[int],
        start_index: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Embed a suffix of a Whisper decoder token history with absolute positions."""
        suffix_tokens = token_history[start_index:]
        if not suffix_tokens:
            raise ValueError("Cannot embed an empty Whisper token suffix.")
        input_ids = torch.tensor([suffix_tokens], dtype=torch.long, device=device)
        position_ids = torch.arange(start_index, len(token_history), dtype=torch.long, device=device).unsqueeze(0)
        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(input_ids, past_key_values_length=start_index, position_ids=position_ids)
        return (inputs_embeds + positions.to(inputs_embeds.device)).unsqueeze(1)

    def _normalize_decoder_logits(
        self,
        logits: torch.Tensor,
        *,
        sequence_length: int,
    ) -> torch.Tensor:
        """Normalize raw Whisper decoder logits to ``(batch, 1, seq_len, vocab)``."""
        if logits.ndim == 2 and int(logits.shape[0]) == sequence_length:
            return logits.reshape(1, 1, sequence_length, -1)
        if logits.ndim == 3:
            return logits.unsqueeze(0)
        return logits

    def decoder_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Union[MobilintWhisperCache, None],
        cache_position: torch.Tensor,
        input_ids: Union[torch.LongTensor, None] = None,
    ) -> torch.Tensor:
        """Run Whisper decoder MXQ by replaying only beam suffixes missing from active cache."""
        batch_size = int(hidden_states.shape[0])
        if past_key_values is None:
            return super().decoder_forward(hidden_states, encoder_hidden_states, past_key_values, cache_position)

        if input_ids is None:
            raise ValueError("MobilintWhisperCache requires input_ids to track beam token histories.")

        input_ids = input_ids.view(batch_size, -1)

        past_key_values.ensure_batch_size(batch_size)
        source_indices = [0] * batch_size
        if encoder_hidden_states.shape[0] == 1 and batch_size > 1:
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, *encoder_hidden_states.shape[1:])
        elif encoder_hidden_states.shape[0] == batch_size:
            source_indices = list(range(batch_size))

        logits_list: list[torch.Tensor] = []
        logits_by_target_tokens: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        mxq_model = self.npu_backend.mxq_model
        trace_enabled = is_whisper_beam_debug_trace_enabled()
        for beam_index in range(batch_size):
            target_tokens = past_key_values.build_target_tokens(beam_index, input_ids[beam_index])
            source_index = source_indices[beam_index]
            target_key = (source_index, tuple(target_tokens))
            prefix_length = past_key_values.get_common_prefix_length(target_tokens)
            if trace_enabled:
                append_whisper_beam_debug_event(
                    {
                        "event": "decoder_beam_before",
                        "beam_index": beam_index,
                        "source_index": source_index,
                        "input_ids": [int(token) for token in input_ids[beam_index].detach().cpu().reshape(-1).tolist()],
                        "target_tokens": list(target_tokens),
                        "active_tokens": past_key_values.get_active_tokens(),
                        "stored_beam_tokens": past_key_values.get_beam_tokens(beam_index),
                        "prefix_length": int(prefix_length),
                    }
                )
            if prefix_length == len(target_tokens) and target_key in logits_by_target_tokens:
                logits_list.append(logits_by_target_tokens[target_key])
                past_key_values.commit_beam_tokens(beam_index, target_tokens)
                if trace_enabled:
                    append_whisper_beam_debug_event(
                        {
                            "event": "decoder_beam_duplicate_logits",
                            "beam_index": beam_index,
                            "source_index": source_index,
                            "target_tokens": list(target_tokens),
                        }
                    )
                continue
            if prefix_length == len(target_tokens):
                prefix_length = max(0, len(target_tokens) - int(input_ids.shape[1]))
            if os.environ.get("MBLT_WHISPER_BEAM_FORCE_FULL_REPLAY"):
                prefix_length = 0
            suffix_hidden_states = self._embed_token_suffix(
                target_tokens,
                prefix_length,
                device=hidden_states.device,
            )
            suffix_length = int(suffix_hidden_states.shape[2])
            hidden_states_numpy = suffix_hidden_states.type(torch.float32).cpu().numpy()
            encoder_hidden_states_numpy = (
                encoder_hidden_states[beam_index : beam_index + 1].type(torch.float32).cpu().numpy()
            )
            result = mxq_model.infer([hidden_states_numpy, encoder_hidden_states_numpy], None, prefix_length)
            assert result is not None, "mxq infer result is None!"
            logits = torch.tensor(result[0], dtype=hidden_states.dtype, device=hidden_states.device)
            logits = self._normalize_decoder_logits(logits, sequence_length=suffix_length)
            current_logits = logits[:, :, -input_ids.shape[1] :, :]
            logits_list.append(current_logits)
            logits_by_target_tokens[target_key] = current_logits
            past_key_values.commit_beam_tokens(beam_index, target_tokens)
            past_key_values.commit_active_tokens(target_tokens)
            if trace_enabled:
                top_values, top_indices = torch.topk(
                    current_logits[:, :, -1, :],
                    k=min(5, current_logits.shape[-1]),
                    dim=-1,
                )
                append_whisper_beam_debug_event(
                    {
                        "event": "decoder_beam_after",
                        "beam_index": beam_index,
                        "source_index": source_index,
                        "target_tokens": list(target_tokens),
                        "prefix_length": int(prefix_length),
                        "suffix_tokens": list(target_tokens[prefix_length:]),
                        "suffix_length": suffix_length,
                        "top_token_ids": [int(token) for token in top_indices.detach().cpu().reshape(-1).tolist()],
                        "top_logits": [float(value) for value in top_values.detach().cpu().reshape(-1).tolist()],
                        "active_tokens": past_key_values.get_active_tokens(),
                    }
                )

        return torch.cat(logits_list, dim=0)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if encoder_hidden_states is not None and encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        if use_cache and past_key_values is None:
            past_key_values = MobilintWhisperCache(self.get_mxq_model(), batch_size=input_shape[0])

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        
        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")

        lm_logits = self.decoder_forward(
            hidden_states.unsqueeze(1),
            cast(torch.Tensor, encoder_hidden_states),
            past_key_values,
            cache_position,
            input_ids=input_ids,
        )

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [lm_logits, next_cache, None, None, None]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=cast(torch.FloatTensor, lm_logits),
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

class MobilintWhisperModel(PretrainedOnlyMixin, MobilintWhisperPreTrainedModel):
    def __init__(self, config: MobilintWhisperConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.encoder = MobilintWhisperEncoder(config, _internal_call=True)
        self.decoder = MobilintWhisperDecoder(config, _internal_call=True)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.get_input_embeddings()
    
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Union[torch.LongTensor, None] = None,
    ):
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(
        self,
        input_features: Union[torch.FloatTensor, None] = None,
        attention_mask: Union[torch.LongTensor, None] = None,
        decoder_input_ids: Union[torch.LongTensor, None] = None,
        decoder_attention_mask: Union[torch.LongTensor, None] = None,
        encoder_outputs: Union[tuple[torch.FloatTensor], BaseModelOutput, None] = None,
        past_key_values: Union[MobilintWhisperCache, None] = None,
        decoder_inputs_embeds: Union[tuple[torch.FloatTensor], None] = None,
        decoder_position_ids: Union[tuple[torch.LongTensor], None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if encoder_outputs is None:
            assert input_features is not None, "input_features is None!"
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            
        assert encoder_outputs is not None, "encoder_outputs is None!"

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        encoder_hidden_states = encoder_outputs[0]
        if encoder_hidden_states.ndim == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        assert isinstance(encoder_outputs, BaseModelOutput), "encoder_outputs is not BaseModelOutput!"

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class MobilintWhisperForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintGenerationMixin,
    WhisperGenerationMixin,
    MobilintWhisperPreTrainedModel,
):
    base_model_prefix = "model"
    
    def __init__(self, config: MobilintWhisperConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)
        
        self.model = MobilintWhisperModel(config, _internal_call=True)
        self.max_target_positions = config.max_target_positions

        # for pipeline type checking
        self.config.model_type = "whisper"
    
    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_cache_mxq_model(self):
        return self.get_decoder().get_mxq_model()

    def _get_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        *args: Any,
    ) -> MobilintWhisperCache:
        """Return a Whisper-specific Mobilint cache for Hugging Face generation."""
        del cache_implementation, max_cache_len, args
        configured_batch_size = max(1, int(batch_size), int(getattr(self.config, "max_batch_size", 1)))
        if not hasattr(self, "_cache"):
            self._cache = MobilintWhisperCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        elif not isinstance(self._cache, MobilintWhisperCache):
            self._cache = MobilintWhisperCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        elif getattr(self._cache, "batch_size", 1) != configured_batch_size:
            self._cache = MobilintWhisperCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        else:
            self._cache.reset()
        return self._cache

    def prepare_inputs_for_generation(
        self,
        *args: Any,
        count_npu_time: bool = False,
        prefill_chunk_size: Union[int, None] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare Whisper generation inputs while preserving Mobilint kwargs."""
        return super().prepare_inputs_for_generation(
            *args,
            count_npu_time=count_npu_time,
            prefill_chunk_size=prefill_chunk_size,
            **kwargs,
        )

    def forward(
        self,
        input_features: Union[torch.FloatTensor, None] = None,
        attention_mask: Union[torch.LongTensor, None] = None,
        decoder_input_ids: Union[torch.LongTensor, None] = None,
        decoder_attention_mask: Union[torch.LongTensor, None] = None,
        encoder_outputs: Union[tuple[tuple[torch.FloatTensor]], None] = None,
        past_key_values: Union[MobilintWhisperCache, None] = None,
        decoder_inputs_embeds: Union[tuple[torch.FloatTensor], None] = None,
        decoder_position_ids: Union[tuple[torch.LongTensor], None] = None,
        labels: Union[torch.LongTensor, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    "Labels' sequence length "
                    f"{labels.shape[1]} cannot exceed the maximum allowed length of "
                    f"{self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = cast(torch.LongTensor, shift_tokens_right(
                    labels, cast(int, self.config.pad_token_id), cast(int, self.config.decoder_start_token_id)
                ))

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = outputs[0].squeeze(1) # proj_out is performed on MobilintWhisperDecoder.

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = cast(torch.LongTensor, labels.to(lm_logits.device))
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

AutoModel.register(MobilintWhisperConfig, MobilintWhisperForConditionalGeneration)
AutoModelForSpeechSeq2Seq.register(MobilintWhisperConfig, MobilintWhisperForConditionalGeneration)
