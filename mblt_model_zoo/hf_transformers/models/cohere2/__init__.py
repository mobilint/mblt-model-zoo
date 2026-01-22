from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_cohere2 import *
    from .modeling_cohere2 import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintCohere2Config":
        module = importlib.import_module(".configuration_cohere2", __package__)
        return module.MobilintCohere2Config
    
    if name == "MobilintCohere2ForCausalLM":
        module = importlib.import_module(".modeling_cohere2", __package__)
        return module.MobilintCohere2ForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintCohere2Config", "MobilintCohere2ForCausalLM"]