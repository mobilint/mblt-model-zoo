from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen3 import *
    from .modeling_qwen3 import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintQwen3Config":
        module = importlib.import_module(".configuration_qwen3", __package__)
        return module.MobilintQwen3Config
    
    if name == "MobilintQwen3ForCausalLM":
        module = importlib.import_module(".modeling_qwen3", __package__)
        return module.MobilintQwen3ForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintQwen3Config", "MobilintQwen3ForCausalLM"]