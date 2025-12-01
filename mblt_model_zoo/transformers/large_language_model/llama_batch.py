import math
import os
from typing import Optional, Tuple, Union, cast

import maccel
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
from mblt_model_zoo.utils.logging import log_model_details

from ..utils.cache_utils import MobilintCache

logger = logging.get_logger(__name__)


class MobilintLlamaBatchConfig(LlamaConfig):
    model_type = "mobilint-llama-batch"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        num_cores: int = 8,
        max_batch_size: int = 16,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        self.num_cores = num_cores
        self.max_batch_size = max_batch_size

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


class MobilintLlamaBatchForCausalLM(LlamaPreTrainedModel, MobilintGenerationMixin):
    config: MobilintLlamaBatchConfig
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}

    def __init__(self, config: MobilintLlamaBatchConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.gradient_checkpointing = False
        
        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(config.num_cores)
        assert config.name_or_path is not None
        model_path = os.path.join(config.name_or_path, config.mxq_path)
        self.mxq_model = maccel.Model(model_path, mc)
        log_model_details(model_path)
        self.mxq_model.launch(self.acc)
    
    def get_cache_mxq_model(self):
        return self.mxq_model

    def set_decoder(self, decoder):
        raise NotImplementedError("set_decoder not available: self.model is implemented in mxq")

    def get_decoder(self):
        logger.warning("get_decoder not available: self.model is implemented in mxq")
        return None

    def tie_weights(self):
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if logits_to_keep > 1:
            logger.warning("logits_to_keep larger than 1 is not supported: %d" % logits_to_keep)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.FloatTensor = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = MobilintCache(self.get_cache_mxq_model())

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        inputs_embeds_numpy: np.ndarray = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds_numpy.ndim == 3:
            # (batch, 1, seqlen, hidden_size)
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)

        # max width should be appropriate number for chunking (ex. 192 for Llama 3.2 3B)
        # it should be searched experimentally
        if chunk_size == 0:
            chunk_size = self.mxq_model.get_input_buffer_info()[0].max_width
        num_of_chunks = math.ceil(inputs_embeds_numpy.shape[2] / chunk_size)

        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds_numpy.shape[2])
            cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()
            batch_param = maccel.BatchParam(
                sequence_lengths=[end_index - start_index],
                cache_sizes=[cache_size],
                cache_ids=[0],
                prefill_masks=[False], # not implemented in C++ yet.
            )

            outputs = self.mxq_model.infer([inputs_embeds_numpy[:, :, start_index:end_index, :]], None, 0, batch_param)

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])
        
        assert outputs is not None
        logits: torch.FloatTensor = cast(torch.FloatTensor, torch.tensor(outputs[0], dtype=torch.float32).squeeze(0))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def launch(self):
        self.get_cache_mxq_model().launch(self.acc)

    def dispose(self):
        self.get_cache_mxq_model().dispose()


AutoConfig.register("mobilint-llama-batch", MobilintLlamaBatchConfig)
AutoModel.register(MobilintLlamaBatchConfig, MobilintLlamaBatchForCausalLM)
AutoTokenizer.register(MobilintLlamaBatchConfig, fast_tokenizer_class=LlamaTokenizerFast)
AutoModelForCausalLM.register(MobilintLlamaBatchConfig, MobilintLlamaBatchForCausalLM)

from ..utils.types import TransformersModelInfo

Llama_32_3B_Instruct_Batch16 = TransformersModelInfo(
    original_model_id="meta-llama/Llama-3.2-3B-Instruct",
    model_id="mobilint/Llama-3.2-3B-Instruct-Batch16",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Llama-3.2-3B-Instruct-Batch16/",
    file_list=[
        "config.json",
        "generation_config.json",
        "Llama-3.2-3B-Instruct-Batch.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)

Llama_31_8B_Instruct_Batch16 = TransformersModelInfo(
    original_model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_id="mobilint/Llama-3.1-8B-Instruct-Batch16",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Llama-3.1-8B-Instruct-Batch16/",
    file_list=[
        "config.json",
        "generation_config.json",
        "Llama-3.1-8B-Instruct-Batch.mxq",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
