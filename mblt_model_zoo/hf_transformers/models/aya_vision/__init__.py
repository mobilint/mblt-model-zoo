from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_aya_vision import *
    from .modeling_aya_vision import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintAyaVisionConfig":
        module = importlib.import_module(".configuration_aya_vision", __package__)
        return module.MobilintAyaVisionConfig
    
    if name == "MobilintAyaVisionForCausalLM":
        module = importlib.import_module(".modeling_aya_vision", __package__)
        return module.MobilintAyaVisionForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintAyaVisionConfig", "MobilintAyaVisionForCausalLM"]