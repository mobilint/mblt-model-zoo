from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_llama_eagle3 import *
    from .modeling_llama_eagle3 import *


def __getattr__(name: str):
    import importlib

    if name in {
        "MobilintLlamaEagle3Config",
    }:
        module = importlib.import_module(".configuration_llama_eagle3", __package__)
        return getattr(module, name)

    if name in {
        "MobilintLlamaEagle3ForCausalLM",
    }:
        module = importlib.import_module(".modeling_llama_eagle3", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MobilintLlamaEagle3Config",
    "MobilintLlamaEagle3ForCausalLM",
]
