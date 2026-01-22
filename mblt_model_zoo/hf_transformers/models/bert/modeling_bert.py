from typing import cast

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertOnlyMLMHead,
    BertPooler,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.modeling_utils import MobilintModelMixin
from .configuration_bert import MobilintBertConfig

logger = logging.get_logger(__name__)

class MobilintBertPreTrainedModel(PreTrainedModel):
    config_class = MobilintBertConfig
    base_model_prefix = "bert"

class MobilintBertModel(MobilintModelMixin, MobilintBertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]
    
    def __init__(self, config: MobilintBertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if use_cache or past_key_values is not None:
            logger.warning("use_cache and past_key_values are not supported.")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if encoder_hidden_states is not None:
            logger.warning("encoder_hidden_states is not supported.")
            
        if encoder_attention_mask is not None:
            logger.warning("encoder_attention_mask is not supported.")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )
        
        sequence_output = self.bert_forward(embedding_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=cast(torch.FloatTensor, sequence_output),
            pooler_output=pooled_output,
            past_key_values=None,
        )
    

class MobilintBertForMaskedLM(MobilintBertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }
    
    def __init__(self, config: MobilintBertConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MobilintBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": None}

    @classmethod
    def can_generate(cls) -> bool:
        """
        Legacy correction: BertForMaskedLM can't call `generate()` from `GenerationMixin`, even though it has a
        `prepare_inputs_for_generation` method.
        """
        return False

    def launch(self):
        self.bert.launch()
    
    def dispose(self):
        self.bert.dispose()
        
AutoModel.register(MobilintBertConfig, MobilintBertForMaskedLM)
AutoModelForMaskedLM.register(MobilintBertConfig, MobilintBertForMaskedLM)
