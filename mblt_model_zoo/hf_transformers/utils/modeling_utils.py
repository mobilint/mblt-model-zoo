import math
import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

from ..utils.cache_utils import MobilintCache
from .base_utils import MobilintNPUBackend, PretrainedOnlyMixin
from .configuration_utils import MobilintConfigMixin, MobilintEncoderDecoderConfigMixin


class MobilintModelMixin(PretrainedOnlyMixin, PreTrainedModel):
    npu_backend_prefix: Literal["", "encoder_", "decoder_"] = ""
    
    def __init__(self, config: MobilintConfigMixin | MobilintEncoderDecoderConfigMixin, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        if TYPE_CHECKING:
            self.config = config
        
        assert self.config.name_or_path is not None, "config.name_or_path is None!"
        
        self.npu_backend: MobilintNPUBackend = self.config.__getattribute__(self.npu_backend_prefix + "npu_backend")
        self.npu_backend.name_or_path = self.config.name_or_path
        self.launch()
    
    def launch(self):
        self.npu_backend.launch()
    
    def dispose(self):
        self.npu_backend.dispose()
    
    def get_mxq_model(self):
        return self.npu_backend.mxq_model

    def mxq_forward(
        self,
        input: torch.Tensor,
    ):
        input_numpy = input.type(torch.float32).cpu().numpy()
        
        result = self.npu_backend.mxq_model.infer([input_numpy])
        assert result is not None, "mxq infer result is None!"

        output = torch.tensor(result[0], dtype=input.dtype, device=input.device)
        
        return output

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
        chunk_size: int = 128,
    ):
        inputs_embeds_numpy = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)  # (batch, 1, seqlen, hidden_size)

        assert chunk_size > 0, "chunk_size should be a positive number! chunk_size: %d" % chunk_size
        num_of_chunks = math.ceil(inputs_embeds_numpy.shape[2] / chunk_size)

        mxq_model = self.npu_backend.mxq_model
        
        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds_numpy.shape[2])
            cache_size = (0 if past_key_values is None else past_key_values.get_seq_length())

            result = mxq_model.infer([inputs_embeds_numpy[:, :, start_index:end_index, :]], None, cache_size)
            assert result is not None, "mxq infer result is None!"
            logits_ndarray = result[0]

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        logits = torch.tensor(logits_ndarray, dtype=inputs_embeds.dtype, device=inputs_embeds.device).squeeze(0)
        
        return logits

    def decoder_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
    ):
        hidden_states_numpy = hidden_states.type(torch.float32).cpu().numpy()
        encoder_hidden_states_numpy = encoder_hidden_states.type(torch.float32).cpu().numpy()

        mxq_model = self.npu_backend.mxq_model
        
        cache_size = (0 if past_key_values is None else past_key_values.get_seq_length())

        result = mxq_model.infer([hidden_states_numpy, encoder_hidden_states_numpy], None, cache_size)
        assert result is not None, "mxq infer result is None!"
        logits_ndarray = result[0]

        if past_key_values is not None:
            past_key_values.update_cache_position(cache_position)

        logits = torch.tensor(logits_ndarray, dtype=hidden_states.dtype, device=hidden_states.device)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        embedding_weight_path = kwargs.pop("embedding_weight", None)
        if embedding_weight_path:
            cls._inject_custom_embeddings(model, embedding_weight_path)

        return model

    @staticmethod
    def _inject_custom_embeddings(model: PreTrainedModel, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom embedding path not found: {path}")

        print(f"[Mobilint] Loading custom embeddings from: {path}")
        
        custom_data = torch.load(path, map_location="cpu")
        
        # Handle dict (state_dict) vs Tensor
        if isinstance(custom_data, dict):
            # Try to find common keys for weights
            if "weight" in custom_data:
                new_weight = custom_data["weight"]
            else:
                # If ambiguous, take the first value
                new_weight = next(iter(custom_data.values()))
        elif isinstance(custom_data, torch.Tensor):
            new_weight = custom_data
        else:
            raise ValueError(f"Unsupported data format in {path}. Expected dict or Tensor.")

        input_embeddings = model.get_input_embeddings()
        
        original_vocab_size = input_embeddings.weight.shape[0]
        new_vocab_size = new_weight.shape[0]
        embed_dim = input_embeddings.weight.shape[1]

        if new_weight.shape[1] != embed_dim:
            raise ValueError(f"Embedding dimension mismatch! Model expects {embed_dim}, but file has {new_weight.shape[1]}")

        if original_vocab_size != new_vocab_size:
            raise ValueError(f"Vocab size mismatch! Model expects {original_vocab_size}, but file has {new_vocab_size}")

        with torch.no_grad():
            input_embeddings.weight.data = new_weight.to(
                device=input_embeddings.weight.device,
                dtype=input_embeddings.weight.dtype
            )
        
        print("[Mobilint] Custom embeddings successfully injected.")
