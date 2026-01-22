from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_exaone import *
    from .modeling_exaone import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintExaoneConfig":
        module = importlib.import_module(".configuration_exaone", __package__)
        return module.MobilintExaoneConfig
    
    if name == "MobilintExaoneForCausalLM":
        module = importlib.import_module(".modeling_exaone", __package__)
        return module.MobilintExaoneForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintExaoneConfig", "MobilintExaoneForCausalLM"]