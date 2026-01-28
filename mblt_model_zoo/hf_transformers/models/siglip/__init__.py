from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_siglip import *
    from .modeling_siglip import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintSiglipVisionConfig":
        module = importlib.import_module(".configuration_siglip", __package__)
        return module.MobilintSiglipVisionConfig
    
    if name == "MobilintSiglipForConditionalGeneration":
        module = importlib.import_module(".modeling_siglip", __package__)
        return module.MobilintSiglipForConditionalGeneration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintSiglipVisionConfig", "MobilintSiglipForConditionalGeneration"]
