import inspect
from typing import Any, get_args

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.cohere2.configuration_cohere2 import Cohere2Config

from ...utils.configuration_utils import MobilintConfigMixin


def _annotation_contains(annotation: Any, expected_type: type) -> bool:
    """Return whether a signature annotation accepts the given type."""
    if annotation is inspect.Signature.empty:
        return False
    if annotation is expected_type:
        return True
    return any(_annotation_contains(arg, expected_type) for arg in get_args(annotation))


_USE_CACHE_PARAMETER = inspect.signature(Cohere2Config.__init__).parameters.get("use_cache")
_USE_CACHE_REQUIRES_INT_COMPAT = bool(_USE_CACHE_PARAMETER) and (
    _annotation_contains(_USE_CACHE_PARAMETER.annotation, int)
    and not _annotation_contains(_USE_CACHE_PARAMETER.annotation, bool)
)


class MobilintCohere2Config(MobilintConfigMixin, Cohere2Config):
    model_type = "mobilint-cohere2"

    def __init__(self, **kwargs: Any) -> None:
        """Normalize `use_cache` for upstream Cohere2 config compatibility."""
        if _USE_CACHE_REQUIRES_INT_COMPAT:
            if "use_cache" not in kwargs:
                kwargs["use_cache"] = 1
            elif isinstance(kwargs["use_cache"], bool):
                kwargs["use_cache"] = int(kwargs["use_cache"])

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


AutoConfig.register("mobilint-cohere2", MobilintCohere2Config)
