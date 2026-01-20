import math
import os
from typing import Optional, Tuple, Union

import maccel
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2PreTrainedModel,
    Qwen2TokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
from mblt_model_zoo.utils.logging import log_model_details

from ..utils.cache_utils import MobilintCache

logger = logging.get_logger(__name__)


class MobilintQwen2Config(Qwen2Config):
    model_type = "mobilint-qwen2"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(tie_word_embeddings=False, **kwargs)


class MobilintQwen2ForCausalLM(Qwen2PreTrainedModel, MobilintGenerationMixin):
    config: MobilintQwen2Config
    supports_gradient_checkpointing = False
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}

    def __init__(self, config: MobilintQwen2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.gradient_checkpointing = False

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(1)
        model_path = os.path.join(config.name_or_path, config.mxq_path)
        self.mxq_model = maccel.Model(model_path, mc)
        log_model_details(model_path)
        self.mxq_model.launch(self.acc)
    
    def get_cache_mxq_model(self):
        return self.mxq_model

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
            logger.warning(f"logits_to_keep larger than 1 is not supported: {logits_to_keep}")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = MobilintCache(self.mxq_model)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        inputs_embeds = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds.ndim == 3:
            inputs_embeds = np.expand_dims(inputs_embeds, 1)  # (batch, 1, seqlen, hidden_size)

        # max width should be appropriate number for chunking (ex. 192 for Qwen2 3.2 3B)
        # it should be searched experimentally
        if chunk_size == 0:
            chunk_size = self.mxq_model.get_input_buffer_info()[0].max_width
        num_of_chunks = math.ceil(inputs_embeds.shape[2] / chunk_size)

        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds.shape[2])
            cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()

            # last infer
            if i == num_of_chunks - 1:
                logits = self.mxq_model.infer([inputs_embeds[:, :, start_index:end_index, :]], None, cache_size)[0]
            else:
                logits = self.mxq_model.infer([inputs_embeds[:, :, start_index:end_index, :]], None, cache_size)[0]

            if use_cache:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        logits = torch.tensor(logits, dtype=torch.float32, device=self.device).squeeze(0)

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
        self.mxq_model.launch(self.acc)

    def dispose(self):
        self.mxq_model.dispose()


AutoConfig.register("mobilint-qwen2", MobilintQwen2Config)
AutoModel.register(MobilintQwen2Config, MobilintQwen2ForCausalLM)
AutoTokenizer.register(MobilintQwen2Config, fast_tokenizer_class=Qwen2TokenizerFast)
AutoModelForCausalLM.register(MobilintQwen2Config, MobilintQwen2ForCausalLM)

from ..utils.types import TransformersModelInfo

Qwen_25_05B_Instruct = TransformersModelInfo(
    original_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_id="mobilint/Qwen2.5-0.5B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Qwen2.5-0.5B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "Qwen2.5-0.5B-Instruct.mxq",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

Qwen_25_15B_Instruct = TransformersModelInfo(
    original_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    model_id="mobilint/Qwen2.5-1.5B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Qwen2.5-1.5B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "Qwen2.5-1.5B-Instruct.mxq",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

Qwen_25_3B_Instruct = TransformersModelInfo(
    original_model_id="Qwen/Qwen2.5-3B-Instruct",
    model_id="mobilint/Qwen2.5-3B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Qwen2.5-3B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "Qwen2.5-3B-Instruct.mxq",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

Qwen_25_7B_Instruct = TransformersModelInfo(
    original_model_id="Qwen/Qwen2.5-7B-Instruct",
    model_id="mobilint/Qwen2.5-7B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/Qwen2.5-7B-Instruct/",
    file_list=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "Qwen2.5-7B-Instruct.mxq",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)
