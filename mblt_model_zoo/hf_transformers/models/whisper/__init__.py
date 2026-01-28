from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_whisper import *
    from .modeling_whisper import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintWhisperConfig":
        module = importlib.import_module(".configuration_whisper", __package__)
        return module.MobilintWhisperConfig
    
    if name == "MobilintWhisperForConditionalGeneration":
        module = importlib.import_module(".modeling_whisper", __package__)
        return module.MobilintWhisperForConditionalGeneration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintWhisperConfig", "MobilintWhisperForConditionalGeneration"]