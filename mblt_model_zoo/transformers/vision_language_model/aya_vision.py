from typing import Optional, Union, List, Tuple, Any, TypeVar

import maccel
from maccel import Cluster, Core, CoreId
import torch
import torch.nn as nn
import numpy as np

from transformers import (
    PretrainedConfig,
    AyaVisionConfig,
    AyaVisionPreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    CohereTokenizerFast,
    AutoProcessor,
    AyaVisionProcessor,
    AutoModelForImageTextToText,
)
from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin

from ..utils.cache_utils import MobilintCache


# type hinting: specifying the type of config class that inherits from PretrainedConfig
SpecificPretrainedConfigType = TypeVar("SpecificPretrainedConfigType", bound="PretrainedConfig")


class MobilintAyaVisionConfig(AyaVisionConfig):
    model_type = "mobilint-aya_vision"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        vision_config = None,
        text_config = None,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        
        if text_config is None:
            text_config = {}
            text_config["model_type"] = "mobilint-cohere2"

        super().__init__(vision_config, text_config, **kwargs)

        if self.vision_feature_select_strategy != "full":
            raise ValueError(
                "vision_feature_select_strategy should be 'full'."
                f"Got: {self.vision_feature_select_strategy}"
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


class MobilintAyaVisionForConditionalGeneration(
    AyaVisionPreTrainedModel, MobilintGenerationMixin
):
    config_class = MobilintAyaVisionConfig

    def __init__(self, config: MobilintAyaVisionConfig):
        super().__init__(config)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.post_init()

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(
            core_ids=[
                CoreId(Cluster.Cluster0, Core.Core2),
                CoreId(Cluster.Cluster0, Core.Core3),
                CoreId(Cluster.Cluster1, Core.Core0),
                CoreId(Cluster.Cluster1, Core.Core1),
                CoreId(Cluster.Cluster1, Core.Core2),
                CoreId(Cluster.Cluster1, Core.Core3),
            ]
        )
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)
    
    def get_mxq_model(self):
        return self.language_model.get_mxq_model()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        if vision_feature_select_strategy != "full":
            raise ValueError(
                f"Unexpected vision_feature_select_strategy: {vision_feature_select_strategy}"
            )

        if vision_feature_layer != -1:
            raise ValueError(f"Unexpected vision_feature_layer: {vision_feature_layer}")

        image_features = pixel_values.type(torch.float32).cpu().numpy()
        image_features = self.mxq_model.infer([image_features])[0]
        image_features = np.transpose(image_features, (0, 2, 3, 1))
        image_features = torch.tensor(image_features, dtype=torch.float32, device=self.device)
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, AyaVisionCausalLMOutputWithPast]:
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
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AyaVisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def dispose(self):
        self.mxq_model.dispose()


AutoConfig.register("mobilint-aya_vision", MobilintAyaVisionConfig)
AutoTokenizer.register(
    MobilintAyaVisionConfig, fast_tokenizer_class=CohereTokenizerFast
)
AutoProcessor.register(MobilintAyaVisionConfig, AyaVisionProcessor)
AutoModelForImageTextToText.register(
    MobilintAyaVisionConfig, MobilintAyaVisionForConditionalGeneration
)

from ..utils.types import TransformersModelInfo

aya_vision_8b = TransformersModelInfo(
    original_model_id="CohereLabs/aya-vision-8b",
    model_id="mobilint/aya-vision-8b",
    download_url_base="https://dl.mobilint.com/model/transformers/vlm/aya-vision-8b/",
    file_list=[
        "c4ai-command-r7b-12-2024.mxq",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "siglip2-so400m-patch14-384-vision.mxq",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
