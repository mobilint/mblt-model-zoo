from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_exaone4 import *
    from .modeling_exaone4 import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintExaone4Config":
        module = importlib.import_module(".configuration_exaone4", __package__)
        return module.MobilintExaone4Config
    
    if name == "MobilintExaone4ForCausalLM":
        module = importlib.import_module(".modeling_exaone4", __package__)
        return module.MobilintExaone4ForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintExaone4Config", "MobilintExaone4ForCausalLM"]