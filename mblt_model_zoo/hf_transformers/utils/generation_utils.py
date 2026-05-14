from abc import ABC
import inspect
from functools import wraps
from typing import Callable, Dict

import qbruntime
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel

from ..utils.cache_utils import MobilintCache
from ..utils.modeling_utils import MobilintModelMixin


def with_mobilint_generation_signature(wrapped: Callable, *extra_keyword_names: str) -> Callable:
    """Preserve an upstream generation hook signature while adding Mobilint kwargs.

    Args:
        wrapped: Upstream callable whose public signature should be preserved.
        *extra_keyword_names: Keyword-only parameters to append before ``**kwargs``.

    Returns:
        Decorator that applies ``functools.wraps`` and exposes an augmented signature.
    """

    def decorator(wrapper: Callable) -> Callable:
        wrapper = wraps(wrapped)(wrapper)
        signature = inspect.signature(wrapped)
        parameters = list(signature.parameters.values())
        existing = set(signature.parameters)
        insert_at = next(
            (idx for idx, parameter in enumerate(parameters) if parameter.kind == inspect.Parameter.VAR_KEYWORD),
            len(parameters),
        )
        for name in extra_keyword_names:
            if name in existing:
                continue
            parameters.insert(
                insert_at,
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool),
            )
            insert_at += 1
        wrapper.__signature__ = signature.replace(parameters=parameters)
        return wrapper

    return decorator


class MobilintGenerationMixin(ABC, GenerationMixin):
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
        configured_batch_size = max(1, getattr(self.config, "max_batch_size", 1))
        if not hasattr(self, "_cache"):
            self._cache = MobilintCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        elif getattr(self._cache, "batch_size", 1) != configured_batch_size:
            self._cache = MobilintCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
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
        elif isinstance(model_kwargs[cache_name], MobilintCache):
            return True
        else:
            model_kwargs[cache_name] = self._get_cache("mobilint", batch_size, max_cache_length, *args, model_kwargs)
            return True
