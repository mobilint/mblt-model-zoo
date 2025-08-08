from typing import Any, Dict, Optional, Tuple

import maccel
import torch

from transformers.cache_utils import CacheLayerMixin, Cache


class MobilintLayer(CacheLayerMixin):
    is_sliding = False
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("update is not implemented")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.model.get_input_buffer_info()[0].max_cache_size

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")
    
    def update_cache_position(self, cache_position: torch.LongTensor):
        self._seen_tokens += cache_position.numel()    
    
class MobilintCache(Cache):
    def __init__(self, model: maccel.Model, *args, **kwargs):
        super().__init__(layer_classes=MobilintLayer, *args, **kwargs)
        self.model = model
        self.model.reset_cache_memory()
        self._seen_tokens: int = 0