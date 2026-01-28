from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_llama import *
    from .modeling_llama import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintLlamaConfig":
        module = importlib.import_module(".configuration_llama", __package__)
        return module.MobilintLlamaConfig
    
    if name == "MobilintLlamaForCausalLM":
        module = importlib.import_module(".modeling_llama", __package__)
        return module.MobilintLlamaForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintLlamaConfig", "MobilintLlamaForCausalLM"]