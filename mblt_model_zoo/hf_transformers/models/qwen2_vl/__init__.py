from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen2_vl import *
    from .modeling_qwen2_vl import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintQwen2VLConfig":
        module = importlib.import_module(".configuration_qwen2_vl", __package__)
        return module.MobilintQwen2VLConfig
    
    if name == "MobilintQwen2VLForConditionalGeneration":
        module = importlib.import_module(".modeling_qwen2_vl", __package__)
        return module.MobilintQwen2VLForConditionalGeneration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintQwen2VLConfig", "MobilintQwen2VLForConditionalGeneration"]