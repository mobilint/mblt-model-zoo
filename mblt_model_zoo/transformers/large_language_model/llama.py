from typing import Dict, Optional, Tuple, TypeVar, Union

import maccel
import torch
import torch.nn as nn
import numpy as np
import math

from transformers import (
    LlamaPreTrainedModel,
    GenerationMixin,
    LlamaConfig,
    LlamaTokenizerFast,
    GenerationConfig,
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import LossKwargs, logging
from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


SpecificPreTrainedModelType = TypeVar(
    "SpecificPreTrainedModelType", bound="PreTrainedModel"
)


class MobilintLlamaConfig(LlamaConfig):
    model_type = "mobilint-llama"

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


class MobilintLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_quantized_cache = False
    _supports_static_cache = False

    config_class = MobilintLlamaConfig

    def __init__(self, config: MobilintLlamaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.gradient_checkpointing = False

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(1)
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        logger.warning_once("self.lm_head is implemented in mxq")
        return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("self.lm_head is implemented in mxq")

    def set_decoder(self, decoder):
        raise NotImplementedError("self.model is implemented in mxq")

    def get_decoder(self):
        logger.warning_once("self.model is implemented in mxq")
        return None

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
        elif model_kwargs[cache_name].__class__.__name__ == "DynamicCache":
            model_kwargs[cache_name] = MobilintCache(self.mxq_model)
        else:
            raise NotImplementedError(
                f"_prepare_cache_for_generation Cache class {model_kwargs[cache_name].__class__.__name__}, which is not compatible for MobilintCache"
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        chunk_size: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        if num_logits_to_keep > 1:
            logger.warning(
                "num_logits_to_keep larger than 1 is not supported: %d"
                % num_logits_to_keep
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = MobilintCache(self.mxq_model)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )

        inputs_embeds = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds.ndim == 3:
            inputs_embeds = np.expand_dims(
                inputs_embeds, 1
            )  # (batch, 1, seqlen, hidden_size)

        # max width should be appropriate number for chunking (ex. 192 for Llama 3.2 3B)
        # it should be searched experimentally
        if chunk_size == 0:
            chunk_size = self.mxq_model.get_input_buffer_info()[0].max_width
        num_of_chunks = math.ceil(inputs_embeds.shape[2] / chunk_size)

        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds.shape[2])
            cache_size = (
                0 if past_key_values is None else past_key_values.get_seq_length()
            )

            # last infer
            if i == num_of_chunks - 1:
                logits = self.mxq_model.infer(
                    [inputs_embeds[:, :, start_index:end_index, :]], None, cache_size
                )[0]
            else:
                logits = self.mxq_model.infer(
                    [inputs_embeds[:, :, start_index:end_index, :]], None, cache_size
                )[0]

            if use_cache:
                past_key_values.update_cache_position(
                    cache_position[start_index:end_index]
                )

        logits = torch.tensor(logits, dtype=torch.float32).squeeze(0)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


AutoConfig.register("mobilint-llama", MobilintLlamaConfig)
AutoModel.register(MobilintLlamaConfig, MobilintLlamaForCausalLM)
AutoTokenizer.register(MobilintLlamaConfig, fast_tokenizer_class=LlamaTokenizerFast)
AutoModelForCausalLM.register(MobilintLlamaConfig, MobilintLlamaForCausalLM)

from ..utils.types import TransformersModelInfo

Llama_32_1B_Instruct = TransformersModelInfo(
    original_model_id="meta-llama/Llama-3.2-1B-Instruct",
    model_id="mobilint/Llama-3.2-1B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Llama-3.2-1B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "Llama-3.2-1B-Instruct.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)

Llama_32_3B_Instruct = TransformersModelInfo(
    original_model_id="meta-llama/Llama-3.2-3B-Instruct",
    model_id="mobilint/Llama-3.2-3B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Llama-3.2-3B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "Llama-3.2-3B-Instruct.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)

Llama_31_8B_Instruct = TransformersModelInfo(
    original_model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_id="mobilint/Llama-3.1-8B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Llama-3.1-8B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "Llama-3.1-8B-Instruct.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)

HyperCLOVAX_SEED_Text_Instruct_15B = TransformersModelInfo(
    original_model_id="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    model_id="mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/HyperCLOVAX-SEED-Text-Instruct-1.5B/",
    file_list=[
        "config.json",
        "HyperCLOVAX-SEED-Text-Instruct-1.5B.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
