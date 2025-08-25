from typing import Dict
import torch
import maccel
from abc import ABC, abstractmethod
from transformers import Cache, GenerationConfig, GenerationMixin, PreTrainedModel
from mblt_model_zoo.transformers.utils.cache_utils import MobilintCache

class MobilintGenerationMixin(ABC, GenerationMixin):        
    @abstractmethod
    def get_mxq_model(self) -> maccel.Model:
        pass
    
    def _get_cache(
        self, cache_implementation: str, batch_size: int, max_cache_len: int, device: torch.device, model_kwargs
    ) -> Cache:
        if not hasattr(self, "_cache"):
            self._cache = MobilintCache(self.get_mxq_model())
        else:
            self._cache.reset()
            
        return self._cache

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model,
            batch_size,
            max_cache_length,
            device,
        )

        cache_name = "past_key_values"

        if model_kwargs.get(cache_name, None) is None:
            return
        elif model_kwargs[cache_name].__class__.__name__ == "MobilintCache":
            return
        elif model_kwargs[cache_name].__class__.__name__ == "DynamicCache":
            model_kwargs[cache_name] = self._get_cache("mobilint", batch_size, max_cache_length, device)
        else:
            raise NotImplementedError(
                f"_prepare_cache_for_generation Cache class {model_kwargs[cache_name].__class__.__name__}, which is not compatible for MobilintCache"
            )