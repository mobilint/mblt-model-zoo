from typing import Optional, Union

import maccel
import torch

from transformers import (
    Cache,
    BertPreTrainedModel,
    BertForMaskedLM,
    BertConfig,
    BertTokenizerFast,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertOnlyMLMHead
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MobilintBertConfig(BertConfig):
    model_type = "mobilint-bert"

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)


class MobilintBertPreTrainedModel(BertPreTrainedModel):
    config: MobilintBertConfig
    load_tf_weights = None
    supports_gradient_checkpointing = False
    _supports_sdpa = False


class MobilintBertModel(MobilintBertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config: MobilintBertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(1)
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)
        self.mxq_model.reset_cache_memory()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError("_prune_heads is not available since self.encoder is implemented in mxq")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if head_mask is not None:
            logger.warning_once("head_mask is not supported.")

        if encoder_hidden_states is not None:
            logger.warning_once("encoder_hidden_states is not supported.")
            
        if encoder_attention_mask is not None:
            logger.warning_once("encoder_attention_mask is not supported.")
        
        if use_cache:
            logger.warning_once("use_cache is not supported.")

        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")
        
        if cache_position is not None:
            logger.warning_once("cache_position is not supported.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = (
                past_key_values[0][0].shape[-2]
                if not isinstance(past_key_values, Cache)
                else past_key_values.get_seq_length()
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        embedding_output = embedding_output.type(torch.float32).cpu().numpy()

        sequence_output = self.mxq_model.infer([embedding_output])[0]
        sequence_output = torch.tensor(sequence_output, dtype=torch.float32, device=self.device)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + (past_key_values,)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def dispose(self):
        self.mxq_model.dispose()


class MobilintBertForMaskedLM(MobilintBertPreTrainedModel, BertForMaskedLM):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        MobilintBertPreTrainedModel.__init__(self, config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MobilintBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def dispose(self):
        self.bert.dispose()


AutoConfig.register("mobilint-bert", MobilintBertConfig)
AutoModel.register(MobilintBertConfig, MobilintBertForMaskedLM)
AutoTokenizer.register(MobilintBertConfig, fast_tokenizer_class=BertTokenizerFast)
AutoModelForMaskedLM.register(MobilintBertConfig, MobilintBertForMaskedLM)

from ..utils.types import TransformersModelInfo

bert_base_uncased = TransformersModelInfo(
    original_model_id="google-bert/bert-base-uncased",
    model_id="mobilint/bert-base-uncased",
    download_url_base="https://dl.mobilint.com/model/transformers/llm/bert-base-uncased/",
    file_list=[
        "bert-base-uncased.mxq",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
