from abc import ABC, abstractmethod
from typing import Dict, Type

import qbruntime
from transformers import Cache, GenerationConfig, GenerationMixin, PreTrainedModel

from ..utils.cache_utils import MobilintBatchCache, MobilintCache
from ..utils.modeling_utils import MobilintModelMixin


class MobilintGenerationMixin(ABC, GenerationMixin):
    cache_cls: Type[Cache] = MobilintCache
    cache_uses_batch_size: bool = False

    def get_cache_mxq_model(self) -> qbruntime.Model:
        if isinstance(self, MobilintModelMixin):
            return self.get_mxq_model()
        else:
            raise TypeError("mxq_model for cache not found! Class: %s" % self.__class__.__name__)
    
    # Function arguments changed for transformers>=4.56.0
    # args contain device and model_kwargs in transformers<4.56.0
    # args contain only model_kwargs in transformers>=4.56.0
    def _get_cache(
        self, cache_implementation: str, batch_size: int, max_cache_len: int, *args
    ) -> MobilintCache:
        if not hasattr(self, "_cache"):
            mxq_model = self.get_cache_mxq_model()
            if self.cache_uses_batch_size:
                self._cache = self.cache_cls(mxq_model, batch_size)  # type: ignore[misc,call-arg]
            else:
                self._cache = self.cache_cls(mxq_model)  # type: ignore[misc,call-arg]
        else:
            self._cache.reset()
            
        return self._cache

    # Function arguments changed for transformers>=4.56.0
    # args contain device in transformers<4.56.0
    # args empty in transformers>=4.56.0
    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        *args,
    ) -> bool:
        super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model, # type: ignore
            batch_size,
            max_cache_length,
            *args,
        )

        cache_name = "past_key_values"

        if model_kwargs.get(cache_name, None) is None:
            return False
        elif isinstance(model_kwargs[cache_name], self.cache_cls):
            return True
        else:
            model_kwargs[cache_name] = self._get_cache("mobilint", batch_size, max_cache_length, *args, model_kwargs)
            return True

class MobilintBatchGenerationMixin(MobilintGenerationMixin):
    cache_cls: Type[Cache] = MobilintBatchCache
    cache_uses_batch_size: bool = True
