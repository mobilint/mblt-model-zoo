from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen2 import *
    from .modeling_qwen2 import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintQwen2Config":
        module = importlib.import_module(".configuration_qwen2", __package__)
        return module.MobilintQwen2Config
    
    if name == "MobilintQwen2ForCausalLM":
        module = importlib.import_module(".modeling_qwen2", __package__)
        return module.MobilintQwen2ForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintQwen2Config", "MobilintQwen2ForCausalLM"]