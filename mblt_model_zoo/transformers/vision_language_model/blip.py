from typing import Optional, Tuple, TypeVar, Union, List, Any

import maccel
import torch
import numpy as np

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    BlipTextConfig,
    BlipVisionConfig,
    BlipPreTrainedModel,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    AutoProcessor,
    BlipProcessor,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
)
from transformers.models.blip.modeling_blip import (
    BlipForConditionalGenerationModelOutput,
)
from transformers.models.blip.modeling_blip_text import (
    BlipTextPreTrainedModel,
    BlipTextEmbeddings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPooling,
)
from transformers.utils import logging

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)


SpecificPreTrainedModelType = TypeVar(
    "SpecificPreTrainedModelType", bound="PreTrainedModel"
)

# type hinting: specifying the type of config class that inherits from PretrainedConfig
SpecificPretrainedConfigType = TypeVar("SpecificPretrainedConfigType", bound="PretrainedConfig")

class MobilintBlipTextConfig(BlipTextConfig):
    model_type = "mobilint-blip_text_model"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


class MobilintBlipVisionConfig(BlipVisionConfig):
    model_type = "mobilint-blip_vision_model"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


class MobilintBlipConfig(PretrainedConfig):
    model_type = "mobilint-blip"
    sub_configs = {
        "text_config": MobilintBlipTextConfig,
        "vision_config": MobilintBlipVisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info(
                "`text_config` is `None`. Initializing the `MobilintBlipTextConfig` with default values."
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                "`vision_config` is `None`. Initializing the `MobilintBlipVisionConfig` with default values."
            )

        self.text_config = MobilintBlipTextConfig(**text_config)
        self.vision_config = MobilintBlipVisionConfig(**vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MobilintBlipTextConfig,
        vision_config: MobilintBlipVisionConfig,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )
    
    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> SpecificPretrainedConfigType:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config, kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        
        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path
        
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
        

class MobilintBlipPreTrainedModel(BlipPreTrainedModel):
    config_class = MobilintBlipConfig
    supports_gradient_checkpointing = False


class MobilintBlipTextPreTrainedModel(BlipTextPreTrainedModel):
    config_class = MobilintBlipTextConfig


class MobilintBlipTextModel(MobilintBlipTextPreTrainedModel):
    def __init__(self, config: MobilintBlipTextConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = BlipTextEmbeddings(config)

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(
            None, [maccel.CoreId(maccel.Cluster.Cluster1, maccel.Core.Core1)]
        )
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("_prune_heads is not implemented")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_decoder: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
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

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if head_mask is not None:
            logger.warning_once("head_mask is not supported.")

        if encoder_attention_mask is not None:
            logger.warning_once("encoder_attention_mask is not supported.")

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds or encoder_embeds"
            )

        if use_cache or past_key_values is not None:
            if past_key_values is None:
                past_key_values = MobilintCache(self.mxq_model)
            elif not isinstance(past_key_values, MobilintCache):
                logger.warning_once(
                    "Class of past_key_values should be MobilintCache, current: "
                    + past_key_values.__class__.__name__
                )
                past_key_values = MobilintCache(self.mxq_model)

        # past_key_values_length
        past_key_values_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        encoder_hidden_states = (
            encoder_hidden_states.unsqueeze(1).type(torch.float32).cpu().numpy()
        )
        embedding_output = (
            embedding_output.unsqueeze(1).type(torch.float32).cpu().numpy()
        )

        logits = self.mxq_model.infer(
            [encoder_hidden_states, embedding_output],
            cache_size=int(past_key_values_length),
        )[0]
        logits = torch.tensor(logits, dtype=torch.float32, device=self.device).squeeze(0)

        past_key_values.update_cache_position(cache_position)

        if not return_dict:
            return (logits, past_key_values)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=logits,
            pooler_output=None,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


class MobilintBlipTextLMHeadModel(MobilintBlipTextPreTrainedModel, MobilintGenerationMixin):
    def __init__(self, config: MobilintBlipTextConfig):
        super().__init__(config)

        self.bert = MobilintBlipTextModel(config)
        self.label_smoothing = config.label_smoothing
    
    def get_mxq_model(self):
        return self.bert.mxq_model

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.bert.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        logger.warning_once("get_output_embeddings is not implemented")
        return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("set_output_embeddings is not implemented")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = False,
        is_decoder: Optional[bool] = True,
        reduction: Optional[str] = "mean",
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        if reduction != "mean":
            logger.warning_once(
                "reduction except 'mean' is not supported: " + reduction
            )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            cache_position=cache_position,
        )

        prediction_scores = outputs[0]

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        if not return_dict:
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # Overwrite -- hardcoded key return (`is_decoder=True`)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        model_inputs["is_decoder"] = True

        return model_inputs

    def dispose(self):
        self.bert.dispose()


class MobilintBlipVisionModel(MobilintBlipPreTrainedModel):
    main_input_name = "pixel_values"
    config_class = MobilintBlipVisionConfig

    def __init__(self, config: MobilintBlipVisionConfig):
        super().__init__(config)
        self.config = config

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_global_core_mode([maccel.Cluster.Cluster0])
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        if interpolate_pos_encoding is True:
            logger.warning_once("interpolate_pos_encoding is not supported.")

        last_hidden_state = self.mxq_model.infer(
            pixel_values.type(torch.float32).cpu().numpy()
        )[0]
        last_hidden_state = np.transpose(last_hidden_state[:, :, 0], (0, 2, 1))
        last_hidden_state = torch.tensor(last_hidden_state, dtype=torch.float32, device=pixel_values.device)

        if not return_dict:
            return (last_hidden_state,)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )

    def get_input_embeddings(self):
        logger.warning_once("get_input_embeddings is not implemented")
        return None

    def dispose(self):
        self.mxq_model.dispose()


class MobilintBlipForConditionalGeneration(
    MobilintBlipPreTrainedModel, MobilintGenerationMixin
):
    config_class = MobilintBlipConfig
    _tied_weights_keys = []
    main_input_name = "pixel_values"

    def __init__(self, config: MobilintBlipConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.vision_model = MobilintBlipVisionModel(config.vision_config)

        self.text_decoder = MobilintBlipTextLMHeadModel(config.text_config)

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id
    
    def get_mxq_model(self):
        return self.text_decoder.get_mxq_model()

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
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

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[0]

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if not return_dict:
            outputs += vision_outputs
            return tuple(output for output in outputs if output is not None)

        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[0]

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor(
                    [[self.decoder_input_ids, self.config.text_config.eos_token_id]]
                )
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=None,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            **generate_kwargs,
        )

        return outputs

    def dispose(self):
        self.vision_model.dispose()
        self.text_decoder.dispose()


AutoConfig.register("mobilint-blip", MobilintBlipConfig)
AutoTokenizer.register(MobilintBlipConfig, BertTokenizer, BertTokenizerFast)
AutoProcessor.register(MobilintBlipConfig, BlipProcessor)
AutoModelForVision2Seq.register(
    MobilintBlipConfig, MobilintBlipForConditionalGeneration
)
AutoModelForImageTextToText.register(
    MobilintBlipConfig, MobilintBlipForConditionalGeneration
)

from ..utils.types import TransformersModelInfo

blip_image_captioning_large = TransformersModelInfo(
    original_model_id="Salesforce/blip-image-captioning-large",
    model_id="mobilint/blip-image-captioning-large",
    download_url_base="https://dl.mobilint.com/model/transformers/vlm/blip-image-captioning-large/",
    file_list=[
        "blip-image-captioning-large_text_decoder.mxq",
        "blip-image-captioning-large_vision_model.mxq",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
