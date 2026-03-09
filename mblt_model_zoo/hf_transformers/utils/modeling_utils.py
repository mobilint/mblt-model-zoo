import math
import os
import time
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import numpy as np
import qbruntime
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from ...utils.npu_backend import MobilintNPUBackend
from ..utils.cache_utils import MobilintCache
from .base_utils import PretrainedOnlyMixin
from .configuration_utils import MobilintConfigMixin, MobilintEncoderDecoderConfigMixin


class MobilintModelMixin(PretrainedOnlyMixin, PreTrainedModel):
    npu_backend_prefix: Literal["", "encoder_", "decoder_"] = ""
    _DEFAULT_PREFILL_CHUNK_SIZE = 128
    
    def __init__(self, config: Union[MobilintConfigMixin, MobilintEncoderDecoderConfigMixin], *args, **kwargs):
        no_launch = kwargs.pop("no_launch", False)
        
        super().__init__(config, *args, **kwargs)
        
        if TYPE_CHECKING:
            self.config = config
        
        assert self.config.name_or_path is not None, "config.name_or_path is None!"

        # Used for benchmark
        self.npu_time = None
        
        self.npu_backend: MobilintNPUBackend = self.config.__getattribute__(self.npu_backend_prefix + "npu_backend")
        self.npu_backend.name_or_path = self.config.name_or_path
        revision = getattr(self.config, "revision", None)
        if revision:
            self.npu_backend.revision = revision
        commit_hash = getattr(self.config, "_commit_hash", None)
        if commit_hash:
            self.npu_backend._commit_hash = commit_hash
        self.npu_backend.create()
        if no_launch != True:
            self.launch()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        embedding_weight_path = kwargs.pop("embedding_weight", None)
        revision = kwargs.get("revision", None)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if embedding_weight_path:
            cls._inject_custom_embeddings(model, embedding_weight_path)

        if hasattr(model, "npu_backend"):
            if revision is not None:
                setattr(model.npu_backend, "revision", revision)
            commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
            if commit_hash:
                setattr(model.npu_backend, "_commit_hash", commit_hash)

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

        input_embeddings: nn.Embedding = cast(nn.Embedding, model.get_input_embeddings())
        
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
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if attention_mask is not None:
            return self._llm_forward_batch(
                inputs_embeds,
                attention_mask,
                past_key_values,
                chunk_size,
            )

        inputs_embeds_numpy = inputs_embeds.type(torch.float32).cpu().numpy()
        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)  # (batch, 1, seqlen, hidden_size)

        resolved_prefill_chunk_size = self.resolve_prefill_chunk_size(prefill_chunk_size)
        num_of_chunks = math.ceil(inputs_embeds_numpy.shape[2] / resolved_prefill_chunk_size)

        mxq_model = self.npu_backend.mxq_model
        self.npu_time = 0.0 if count_npu_time else None
        
        for i in range(num_of_chunks):
            start_index = i * resolved_prefill_chunk_size
            end_index = min(start_index + resolved_prefill_chunk_size, inputs_embeds_numpy.shape[2])
            cache_size = (0 if past_key_values is None else past_key_values.get_seq_length())

            if count_npu_time:
                t0 = time.perf_counter()
                result = mxq_model.infer([inputs_embeds_numpy[:, :, start_index:end_index, :]], None, cache_size)
                self.npu_time += time.perf_counter() - t0
            else:
                result = mxq_model.infer([inputs_embeds_numpy[:, :, start_index:end_index, :]], None, cache_size)

            assert result is not None, "mxq infer result is None!"
            logits_ndarray = result[0]

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        logits = torch.tensor(logits_ndarray, dtype=inputs_embeds.dtype, device=inputs_embeds.device).squeeze(0)

        return logits
    
    def _llm_forward_batch(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        chunk_size: int = 0,
    ):
        batch_size = attention_mask.shape[0]

        if attention_mask.shape == inputs_embeds.shape[:-1]:
            attention_mask_bool = cast(torch.BoolTensor, attention_mask.type(torch.bool))
            inputs_embeds_masked = [inputs_embeds[i, attention_mask_bool[i, :], :] for i in range(batch_size)]
            sequence_lengths = cast(list[int], attention_mask.sum(dim=1).tolist())
        else:
            assert inputs_embeds.shape[1] == 1
            inputs_embeds_masked = [inputs_embeds[i, :, :] for i in range(batch_size)]
            sequence_lengths = [1 for _ in range(batch_size)]

        max_sequence_length = max(sequence_lengths)
        mxq_model = self.npu_backend.mxq_model
        if chunk_size == 0:
            chunk_size = mxq_model.get_input_buffer_info()[0].max_width
        assert chunk_size > 0, "chunk_size should be a positive number! chunk_size: %d" % chunk_size
        num_of_chunks = math.ceil(max_sequence_length / chunk_size)

        logits_dict: dict[int, torch.Tensor] = {}

        for i in range(num_of_chunks):
            start_index = i * chunk_size

            sequence_lengths_chunks: list[int] = []
            cache_sizes_chunks: list[int] = []
            cache_ids: list[int] = []
            prefill_masks: list[bool] = []
            inputs_embeds_chunks: list[torch.Tensor] = []
            seen_tokens: dict[int, int] = {}

            for j in range(batch_size):
                end_index = min(start_index + chunk_size, sequence_lengths[j])
                if start_index < sequence_lengths[j] and end_index <= sequence_lengths[j]:
                    sequence_lengths_chunks.append(end_index - start_index)
                    cache_sizes_chunks.append(past_key_values.get_seq_length(j) if past_key_values is not None else 0)
                    cache_ids.append(j)
                    prefill_masks.append(end_index < inputs_embeds_masked[j].shape[0])
                    inputs_embeds_chunks.append(inputs_embeds_masked[j][start_index:end_index, :])
                    seen_tokens[j] = end_index - start_index

            if len(inputs_embeds_chunks) == 0:
                continue

            inputs_embeds_concat = torch.concat(inputs_embeds_chunks, dim=0).unsqueeze(0)
            inputs_embeds_numpy: np.ndarray = inputs_embeds_concat.type(torch.float32).cpu().numpy()

            if inputs_embeds_numpy.ndim == 3:
                inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)

            batch_param = qbruntime.BatchParam(
                sequence_lengths=sequence_lengths_chunks,
                cache_sizes=cache_sizes_chunks,
                cache_ids=cache_ids,
                prefill_masks=prefill_masks,
            )
            result = mxq_model.infer([inputs_embeds_numpy], None, 0, batch_param)
            assert result is not None, "mxq infer result is None!"

            logits_chunks = cast(
                torch.FloatTensor,
                torch.tensor(result[0], dtype=torch.float32).reshape([len(cache_ids), 1, -1]),
            )

            for j, prefill_mask in enumerate(prefill_masks):
                if prefill_mask is False:
                    cache_id = cache_ids[j]
                    logits_dict[cache_id] = logits_chunks[j, :, :].clone()

            if past_key_values is not None:
                past_key_values.update_seen_tokens(seen_tokens)

        logits_list = [logits_dict[cache_id] for cache_id in range(batch_size)]
        return cast(torch.FloatTensor, torch.stack(logits_list, dim=0))

    def resolve_prefill_chunk_size(self, prefill_chunk_size: Optional[int]) -> int:
        explicit_prefill_chunk_size = self._coerce_positive_int(prefill_chunk_size)
        if explicit_prefill_chunk_size is not None:
            return explicit_prefill_chunk_size

        config_value = self._get_config_prefill_chunk_size()
        config_prefill_chunk_size = self._coerce_positive_int(config_value)
        if config_prefill_chunk_size is not None:
            return config_prefill_chunk_size

        return self._DEFAULT_PREFILL_CHUNK_SIZE

    def _get_config_prefill_chunk_size(self) -> Any:
        config_value = getattr(self.config, "npu_prefill_chunk_size", None)
        if isinstance(config_value, dict):
            core_mode = getattr(self.npu_backend, "core_mode", None)
            if core_mode is None:
                return None
            return config_value.get(core_mode)
        return config_value

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer():
            return int(value) if value > 0 else None
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        return None

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
