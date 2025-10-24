from typing import Optional, Union

import maccel
import torch
import torch.nn as nn
import numpy as np
import math

from transformers.models.cohere2.configuration_cohere2 import Cohere2Config
from transformers.models.cohere2 import Cohere2PreTrainedModel
from transformers.utils import TransformersKwargs, logging
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    CohereTokenizerFast,
)

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin
from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)


class MobilintCohere2Config(Cohere2Config):
    model_type = "mobilint-cohere2"

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
    
    @property
    def sliding_window_pattern(self):
        # Suppress warning only in transformers==4.54.1
        return self._sliding_window_pattern


class MobilintCohere2ForCausalLM(Cohere2PreTrainedModel, MobilintGenerationMixin):
    config: MobilintCohere2Config
    supports_gradient_checkpointing = False
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}

    def __init__(self, config: MobilintCohere2Config, *inputs, **kwargs):
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

        self.logit_scale = config.logit_scale
        self.tie_word_embeddings = config.tie_word_embeddings
    
    def get_mxq_model(self):
        return self.mxq_model

    def set_decoder(self, decoder):
        raise NotImplementedError("self.model is implemented in mxq")

    def get_decoder(self):
        logger.warning_once("self.model is implemented in mxq")
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        if logits_to_keep > 1:
            logger.warning(
                "logits_to_keep larger than 1 is not supported: %d" % logits_to_keep
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

        logits = torch.tensor(logits, dtype=torch.float32, device=self.device).squeeze(0)
        logits = logits * self.logit_scale  # main diff from Llama

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


AutoConfig.register("mobilint-cohere2", MobilintCohere2Config)
AutoModel.register(MobilintCohere2Config, MobilintCohere2ForCausalLM)
AutoTokenizer.register(MobilintCohere2Config, fast_tokenizer_class=CohereTokenizerFast)
AutoModelForCausalLM.register(MobilintCohere2Config, MobilintCohere2ForCausalLM)

from ..utils.types import TransformersModelInfo

c4ai_command_r7b_12_2024 = TransformersModelInfo(
    original_model_id="CohereLabs/c4ai-command-r7b-12-2024",
    model_id="mobilint/c4ai-command-r7b-12-2024",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/c4ai-command-r7b-12-2024/",
    file_list=[
        "c4ai-command-r7b-12-2024.mxq",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
