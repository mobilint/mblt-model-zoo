from typing import Any, List, Optional

import maccel
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class MobilintLayer(CacheLayerMixin):
    is_sliding = False
    
    def __init__(self, mxq_model: maccel.Model, cache_id: int = 0):
        self.mxq_model = mxq_model
        self.cache_id = cache_id
        self._seen_tokens = 0
        self.buffer: List[bytes] = []
        
    def lazy_initialization(self, key_states: torch.Tensor):
        raise NotImplementedError("lazy_initialization is not implemented")

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("update is not implemented")

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self, cache_position=None) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.mxq_model.get_input_buffer_info()[0].max_cache_size
    
    def reset(self) -> None:
        # self.mxq_model.reset_cache_memory()
        self._seen_tokens: int = 0
        self.buffer = []

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")
    
    def update_cache_position(self, cache_position: torch.Tensor):
        self._seen_tokens += cache_position.numel()
    
    def update_seen_tokens(self, num_new_seen_tokens: int):
        self._seen_tokens += num_new_seen_tokens
    
    def dump_cache_memory(self):
        self.buffer = self.mxq_model.dump_cache_memory(self.cache_id)
    
    def load_cache_memory(self):
        if self.get_seq_length() > 0:
            self.mxq_model.reset_cache_memory()
            self.mxq_model.load_cache_memory(self.buffer, self.cache_id)

class MobilintCache(Cache):
    def __init__(self, mxq_model: maccel.Model):
        self.mxq_model = mxq_model
        self.mxq_model.reset_cache_memory()
        
        self.layers: list[MobilintLayer] = [MobilintLayer(self.mxq_model)]
        self.layer_classes = MobilintLayer
                
        self.cache_processor = None
    
    def update_cache_position(self, cache_position: torch.Tensor):
        self.layers[0].update_cache_position(cache_position)
    
    def dump_cache_memory(self):
        self.layers[0].dump_cache_memory()
    
    def load_cache_memory(self):
        self.layers[0].load_cache_memory()

class MobilintBatchCache(Cache):
    def __init__(self, mxq_model: maccel.Model, batch_size: int = 16):
        self.mxq_model = mxq_model
        self.mxq_model.reset_cache_memory()
        
        self.batch_size = batch_size
        
        self.layers: list[MobilintLayer] = [MobilintLayer(self.mxq_model, cache_id) for cache_id in range(self.batch_size)]
        self.layer_classes = MobilintLayer
                
        self.cache_processor = None
    
    def get_seq_length(self, index: int = 0) -> int:
        return self.layers[index].get_seq_length()
    
    def update_seen_tokens(self, sequence_lengths: dict[int, int]):
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].update_seen_tokens(seq_len)
    
    def dump_cache_memory(self, cache_id: int):
        self.layers[cache_id].dump_cache_memory()
    
    def load_cache_memory(self, cache_id: int):
        self.layers[cache_id].load_cache_memory()