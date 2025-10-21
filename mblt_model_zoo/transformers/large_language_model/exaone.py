from typing import Optional, Tuple, TypeVar, Union

import maccel
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import CrossEntropyLoss

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.utils import logging

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)

SpecificPreTrainedModelType = TypeVar(
    "SpecificPreTrainedModelType", bound="PreTrainedModel"
)


class ExaoneConfig(PretrainedConfig):
    model_type = "exaone"
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=102400,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=None,
        activation_function="silu",
        rope_theta=10000.0,
        rope_scaling=None,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if intermediate_size:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = hidden_size * 4
        self.activation_function = activation_function
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class MobilintExaoneConfig(ExaoneConfig):
    model_type = "mobilint-exaone"

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


class MobilintExaoneForCausalLM(PreTrainedModel, MobilintGenerationMixin):
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    config_class = MobilintExaoneConfig

    def __init__(self, config: MobilintExaoneConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(
            config.vocab_size, self.embed_dim, self.config.pad_token_id
        )

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(1)
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)
    
    def get_mxq_model(self):
        return self.mxq_model

    def get_output_embeddings(self):
        logger.warning_once("self.lm_head is implemented in mxq")
        return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("self.lm_head is implemented in mxq")

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
        chunk_size: int = 0,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

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

        lm_logits = torch.tensor(logits, dtype=torch.float32, device=self.device).squeeze(0)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(self.config.torch_dtype)
            loss = loss.to(self.config.torch_dtype)

        if not return_dict:
            output = (lm_logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


AutoConfig.register("mobilint-exaone", MobilintExaoneConfig)
AutoModel.register(MobilintExaoneConfig, MobilintExaoneForCausalLM)
AutoTokenizer.register(
    MobilintExaoneConfig,
    fast_tokenizer_class=GPT2TokenizerFast,
    slow_tokenizer_class=GPT2Tokenizer,
)
AutoModelForCausalLM.register(MobilintExaoneConfig, MobilintExaoneForCausalLM)

from ..utils.types import TransformersModelInfo

EXAONE_35_24B_Instruct = TransformersModelInfo(
    original_model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    model_id="mobilint/EXAONE-3.5-2.4B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/EXAONE-3.5-2.4B-Instruct/",
    file_list=[
        "config.json",
        "EXAONE-3.5-2.4B-Instruct.mxq",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

EXAONE_Deep_24B = TransformersModelInfo(
    original_model_id="LGAI-EXAONE/EXAONE-Deep-2.4B",
    model_id="mobilint/EXAONE-Deep-2.4B",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/EXAONE-Deep-2.4B/",
    file_list=[
        "config.json",
        "EXAONE-Deep-2.4B.mxq",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)
