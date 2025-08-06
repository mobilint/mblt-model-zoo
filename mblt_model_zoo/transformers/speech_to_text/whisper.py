from typing import Dict, Optional, Tuple, TypeVar, Union, Callable, List
import math
import warnings
import maccel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np


from transformers import (
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperProcessor,
    GenerationConfig,
    PreTrainedModel,
    WhisperConfig,
    WhisperPreTrainedModel,
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperPositionalEmbedding,
    shift_tokens_right,
)
from transformers.models.whisper.generation_whisper import (
    WhisperGenerationMixin,
    _get_attr_from_logit_processors,
    _pad_to_max_length,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import logging
from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)

SpecificPreTrainedModelType = TypeVar(
    "SpecificPreTrainedModelType", bound="PreTrainedModel"
)


class MobilintWhisperConfig(WhisperConfig):
    model_type = "mobilint-whisper"

    def __init__(
        self,
        encoder_mxq_path: str = "",
        decoder_mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.encoder_mxq_path = encoder_mxq_path
        self.decoder_mxq_path = decoder_mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


class MobilintWhisperPreTrainedModel(WhisperPreTrainedModel):
    config_class = MobilintWhisperConfig
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_static_cache = False

    def _init_weights(self, module):
        raise NotImplementedError("_init_weights is not implemented")


class MobilintWhisperEncoder(MobilintWhisperPreTrainedModel):
    def __init__(self, config: MobilintWhisperConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1_stride = 1
        self.conv2_stride = 2

        self.gradient_checkpointing = False

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_global_core_mode([maccel.Cluster.Cluster1])
        self.mxq_model = maccel.Model(
            f"{config.name_or_path}/{config.encoder_mxq_path}", mc
        )
        self.mxq_model.launch(self.acc)

    def _freeze_parameters(self):
        raise NotImplementedError("_freeze_parameters is not implemented")

    def get_input_embeddings(self) -> nn.Module:
        logger.warning_once("get_input_embeddings is not implemented")
        return None

    def set_input_embeddings(self, value: nn.Module):
        raise NotImplementedError("set_input_embeddings is not implemented")

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = (
            self.config.max_source_positions * self.conv1_stride * self.conv2_stride
        )
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if head_mask is not None:
            logger.warning_once("head_mask is not supported.")

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        output = self.mxq_model.infer(
            input_features.permute(0, 2, 1).type(torch.float32).cpu().numpy()
        )
        hidden_states = torch.from_numpy(output[0]).to("cpu").unsqueeze(0)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=None, attentions=None
        )

    def dispose(self):
        self.mxq_model.dispose()


class MobilintWhisperDecoder(MobilintWhisperPreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config: MobilintWhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model
        )

        self._use_flash_attention_2 = False
        self._use_sdpa = False

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(
            None, [maccel.CoreId(maccel.Cluster.Cluster0, maccel.Core.Core3)]
        )
        self.mxq_model = maccel.Model(
            f"{config.name_or_path}/{config.decoder_mxq_path}", mc
        )
        self.mxq_model.launch(self.acc)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is not None:
            logger.warning_once("attention_mask is not supported.")

        if head_mask is not None:
            logger.warning_once("head_mask is not supported.")

        if cross_attn_head_mask is not None:
            logger.warning_once("cross_attn_head_mask is not supported.")

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache or past_key_values is not None:
            if not isinstance(past_key_values, MobilintCache):
                logger.warning_once(
                    "Class of past_key_values should be MobilintCache, current: "
                    + past_key_values.__class__.__name__
                )
                past_key_values = MobilintCache(self.mxq_model)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + input_shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )
        else:
            positions = self.embed_positions(
                inputs_embeds,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        inputs = [
            encoder_hidden_states.type(torch.float32).cpu().numpy(),
            hidden_states.unsqueeze(0).type(torch.float32).cpu().numpy(),
        ]
        logits = torch.from_numpy(
            self.mxq_model.infer(inputs, cache_size=int(past_key_values_length))[0]
        ).to(self.device)

        if use_cache:
            past_key_values.update_cache_position(cache_position)

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(logits, next_cache)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits,
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


class MobilintWhisperModel(MobilintWhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = MobilintWhisperEncoder(config)
        self.decoder = MobilintWhisperDecoder(config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        self.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[MobilintCache] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            if attention_mask is not None:
                logger.warning_once("attention_mask is not supported.")

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
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

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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

    def dispose(self):
        self.encoder.dispose()
        self.decoder.dispose()


class MobilintWhisperForConditionalGeneration(
    WhisperGenerationMixin, MobilintWhisperPreTrainedModel
):
    base_model_prefix = "model"

    def __init__(self, config: MobilintWhisperConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = MobilintWhisperModel(config)
        self.max_target_positions = config.max_target_positions
        # for pipeline type checking
        self.config.model_type = "whisper"

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        logger.warning_once("get_output_embeddings is not implemented")
        return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("set_output_embeddings is not implemented")

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        self.model.encoder._freeze_parameters()

    def tie_weights(self):
        pass

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model,
            batch_size,
            max_cache_length,
            device,
        )

        cache_name = "past_key_values"

        if model_kwargs.get(cache_name, None) is None:
            return
        elif model_kwargs[cache_name].__class__.__name__ == "MobilintCache":
            return
        elif model_kwargs[cache_name].__class__.__name__ == "EncoderDecoderCache":
            model_kwargs[cache_name] = MobilintCache(self.model.decoder.mxq_model)
        else:
            raise NotImplementedError(
                f"_prepare_cache_for_generation Cache class {model_kwargs[cache_name].__class__.__name__}, which is not compatible for MobilintCache"
            )

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[MobilintCache] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = outputs[0].squeeze(0)  # proj_out is performed on decoder mblt.

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1)
            )

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

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[Union[str, List[str]]] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        time_precision_features: float = 0.01,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        force_unique_generate_call: Optional[bool] = None,
        **kwargs,
    ):
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )

        # 1. prepare generation config
        generation_config, kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1_stride * self.model.encoder.conv2_stride
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        return_dict_in_generate = self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        timestamp_begin = self._set_return_timestamps(
            return_timestamps=return_timestamps,
            is_shortform=is_shortform,
            generation_config=generation_config,
        )
        self._set_language_and_task(
            language=language,
            task=task,
            is_multilingual=is_multilingual,
            generation_config=generation_config,
        )
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps,
            generation_config=generation_config,
            kwargs=kwargs,
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            batch_size=batch_size,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        device = (
            kwargs["encoder_outputs"][0].device
            if "encoder_outputs" in kwargs
            else input_features.device
        )
        begin_index = init_tokens.shape[1]
        num_beams = kwargs.get(
            "num_beams",
            (
                generation_config.num_beams
                if hasattr(generation_config, "num_beams")
                and generation_config.num_beams is not None
                else 1
            ),
        )
        if "assistant_model" in kwargs:
            # speculative decoding: the model should be able to return eos token
            generation_config.begin_suppress_tokens = None

        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            num_beams=num_beams,
            device=device,
        )

        # 4 Set and retrieve global generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens,
            generation_config=generation_config,
        )

        temperatures = (
            [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        )
        temperature = temperatures[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size,
            attention_mask=attention_mask,
            total_input_frames=total_input_frames,
            is_shortform=is_shortform,
        )

        # 5 Prepare running variables, list for generation
        num_return_sequences = generation_config.num_return_sequences
        (
            batch_idx_map,
            cur_bsz,
            input_features,
            seek,
            max_frames,
            init_tokens,
            do_condition_on_prev_tokens,
        ) = self._expand_variables_for_generation(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            init_tokens=init_tokens,
            batch_size=batch_size,
            condition_on_prev_tokens=condition_on_prev_tokens,
            generation_config=generation_config,
        )

        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=cur_bsz,
            generation_config=generation_config,
        )
        # 5bis speculative decoding: ensure the assistant model does only one call to generate and therefore returns decoder input token ids and eos token id
        # we set a flag in the generation config to force the model to make only one call to generate and return the decoder input token ids and eos token id
        if "assistant_model" in kwargs:
            assistant_model = kwargs["assistant_model"]
            assistant_model.generation_config.force_unique_generate_call = True

        if force_unique_generate_call is None:
            if hasattr(generation_config, "force_unique_generate_call"):
                force_unique_generate_call = (
                    generation_config.force_unique_generate_call
                )
            elif hasattr(self.generation_config, "force_unique_generate_call"):
                force_unique_generate_call = (
                    self.generation_config.force_unique_generate_call
                )
            else:
                force_unique_generate_call = False

        # 6 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = (
                seek.to(torch.float32 if device.type == "mps" else torch.float64)
                * time_precision
                / input_stride
            )
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.2 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.3 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )

            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=init_tokens.device,
                suppress_tokens=suppress_tokens,
                timestamp_begin=timestamp_begin,
                kwargs=kwargs,
            )

            # 6.4 set max new tokens or max length
            self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
            )

            # 6.5 Set current `begin_index` for all logit processors
            if logits_processor is not None:
                for proc in logits_processor:
                    if hasattr(proc, "set_begin_index"):
                        proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.6 Run generate with fallback
            (
                seek_sequences,
                seek_outputs,
                should_skip,
                do_condition_on_prev_tokens,
                model_output_type,
            ) = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                is_shortform=is_shortform,
                batch_size=batch_size,
                attention_mask=attention_mask,
                kwargs=kwargs,
            )

            # 6.7 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    time_precision_features=time_precision_features,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                    decoder_input_ids=decoder_input_ids,
                )

                seek[prev_i] += segment_offset

                current_segments[prev_i] += segments

            if force_unique_generate_call:
                break

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (
                prompt_ids is not None
                and generation_config.prompt_condition_type == "first-segment"
            )
            else current_segments
        )

        # if return_dict_in_generate=True and we forced a unique call to generate or return_timestamps=False, meaning we are sure only one call to generate has been made,
        # -> we can return a ModelOutput
        # otherwise, return_dict_in_generate is applied in the 'result' of each segment in final_segments
        if (
            return_dict_in_generate
            and generation_config.return_dict_in_generate
            and (force_unique_generate_call or not return_timestamps)
        ):
            # only one call to generate_with_fallback, we can return a ModelOutput
            outputs = self._stack_split_outputs(
                seek_outputs, model_output_type, self.device, kwargs
            )
            if num_return_sequences > 1:
                if (
                    hasattr(outputs, "encoder_attentions")
                    and outputs.encoder_attentions is not None
                ):
                    outputs.encoder_attentions = tuple(
                        outputs.encoder_attentions[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_attentions))
                    )
                if (
                    hasattr(outputs, "encoder_hidden_states")
                    and outputs.encoder_hidden_states is not None
                ):
                    outputs.encoder_hidden_states = tuple(
                        outputs.encoder_hidden_states[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_hidden_states))
                    )
            return outputs

        padded_outputs = _pad_to_max_length(
            current_segments=final_segments,
            pad_token_id=generation_config.pad_token_id,
            device=self.device,
            padding_side="right",
            return_token_timestamps=return_token_timestamps,
            force_unique_generate_call=force_unique_generate_call,
        )

        if return_dict_in_generate and generation_config.return_dict_in_generate:
            logger.warning_once(
                "You have passed `return_dict_in_generate=True` and `return_timestamps=True`, this automatically sets `return_segments=True` to access the resuls of the underlying calls to GenerationMixin's generate in the returned `segments`."
            )
            return_segments = True
        elif not return_segments and not return_token_timestamps:
            return padded_outputs

        if return_token_timestamps:
            sequences, token_timestamps = padded_outputs
            outputs = {
                "sequences": sequences,
                "token_timestamps": token_timestamps,
            }
        else:
            sequences = padded_outputs
            outputs = {
                "sequences": sequences,
            }

        if return_segments:
            outputs["segments"] = final_segments

        return outputs


AutoConfig.register("mobilint-whisper", MobilintWhisperConfig)
AutoTokenizer.register(MobilintWhisperConfig, WhisperTokenizer)
AutoFeatureExtractor.register(MobilintWhisperConfig, WhisperFeatureExtractor)
AutoProcessor.register(MobilintWhisperConfig, WhisperProcessor)
AutoModelForSpeechSeq2Seq.register(
    MobilintWhisperConfig, MobilintWhisperForConditionalGeneration
)

from ..utils.types import TransformersModelInfo

whisper_small = TransformersModelInfo(
    original_model_id="openai/whisper-small",
    model_id="mobilint/whisper-small",
    download_url_base="https://dl.mobilint.com/model/transformers/stt/whisper-small/",
    file_list=[
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "normalizer.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "whisper-small_encoder.mxq",
        "whisper-small_decoder.mxq",
    ],
)
