from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_blip import *
    from .modeling_blip import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintBlipConfig":
        module = importlib.import_module(".configuration_blip", __package__)
        return module.MobilintBlipConfig
    
    if name == "MobilintBlipForConditionalGeneration":
        module = importlib.import_module(".modeling_blip", __package__)
        return module.MobilintBlipForConditionalGeneration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintBlipConfig", "MobilintBlipForConditionalGeneration"]