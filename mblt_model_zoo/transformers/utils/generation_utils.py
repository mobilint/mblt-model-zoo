import torch
import maccel
from abc import ABC, abstractmethod
from transformers import Cache, GenerationMixin
from mblt_model_zoo.transformers.utils.cache_utils import MobilintCache

class MobilintGenerationMixin(ABC, GenerationMixin):
    @abstractmethod
    def get_mxq_model(self) -> maccel.Model:
        pass
    
    def _get_cache(
        self, cache_implementation: str, batch_size: int, max_cache_len: int, device: torch.device, model_kwargs
    ) -> Cache:
        if self._cache is None:
            self._cache = MobilintCache(self.get_mxq_model())
        else:
            self._cache.reset()
            
        return self._cache