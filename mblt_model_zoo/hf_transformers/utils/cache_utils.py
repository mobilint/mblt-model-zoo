import copy
from pathlib import Path
from typing import Any, List, Optional

import qbruntime
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class MobilintLayer(CacheLayerMixin):
    is_sliding = False
    
    def __init__(self, mxq_model: qbruntime.Model):
        self.mxq_model = mxq_model
        self._seen_tokens = 0
        
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
        self._seen_tokens: int = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")
    
    def update_cache_position(self, cache_position: torch.Tensor):
        self._seen_tokens += cache_position.numel()
    
class MobilintCache(Cache):
    def __init__(self, mxq_model: qbruntime.Model):
        self.mxq_model = mxq_model
        
        self.layers: list[MobilintLayer] = [MobilintLayer(self.mxq_model)]
        self.layer_classes = MobilintLayer
        
        self.num_hidden_layers = 1
        
        self.cache_processor = None

        self.buffer: List[bytes] = []
    
    def reset(self):
        super().reset()
        
        self.buffer = []
    
    def update_cache_position(self, cache_position: torch.Tensor):
        self.layers[0].update_cache_position(cache_position)
    
    def dump_cache_memory(self):
        self.buffer = self.mxq_model.dump_cache_memory()
    
    def load_cache_memory(self):
        if self.get_seq_length() > 0:
            self.mxq_model.load_cache_memory(self.buffer)
    
    def dump_cache_memory_to(self, cache_dir: str):
        self.mxq_model.dump_cache_memory_to(cache_dir)
        seq_path = Path(cache_dir) / "seq_length.txt"
        seq_path.write_text(f"{self.get_seq_length()}\n", encoding="utf-8")
    
    def load_cache_memory_from(self, cache_dir: str):
        self.reset()
        seq_path = Path(cache_dir) / "seq_length.txt"
        if seq_path.exists():
            seq_length = int(seq_path.read_text(encoding="utf-8").strip())
        else:
            seq_length = 0
        self.layers[0]._seen_tokens = seq_length
        self.mxq_model.load_cache_memory_from(cache_dir)

    def copy(self):
        copied = MobilintCache(self.mxq_model)
        copied.layers[0]._seen_tokens = self.layers[0]._seen_tokens
        copied.buffer = copy.deepcopy(self.buffer)
        
        return copied
