from typing import Optional, Tuple, TypeVar, Union
import math
import maccel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


from transformers import (
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperProcessor,
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
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils import logging

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
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

class Object(object):
    pass

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
        
        self.conv1 = lambda: None
        self.conv2 = lambda: None
        
        self.conv1.stride = [1]
        self.conv2.stride = [2]

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
            self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
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
        )[0]
        hidden_states = torch.tensor(output, dtype=torch.float32, device=input_features.device).unsqueeze(0)

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
            hidden_states.unsqueeze(1).type(torch.float32).cpu().numpy(),
        ]
        logits = self.mxq_model.infer(inputs, cache_size=int(past_key_values_length))[0]
        logits = torch.tensor(logits, dtype=torch.float32, device=self.device)

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
    MobilintGenerationMixin, WhisperGenerationMixin, MobilintWhisperPreTrainedModel
):
    base_model_prefix = "model"

    def __init__(self, config: MobilintWhisperConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = MobilintWhisperModel(config)
        self.max_target_positions = config.max_target_positions
        # for pipeline type checking
        self.config.model_type = "whisper"
    
    def get_mxq_model(self):
        return self.model.decoder.mxq_model

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
        lm_logits = outputs[0].squeeze(1)  # proj_out is performed on decoder mblt.

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
    
    def dispose(self):
        self.model.dispose()


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
