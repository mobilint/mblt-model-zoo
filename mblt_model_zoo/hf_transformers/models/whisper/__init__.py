from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_whisper import MobilintWhisperConfig
    from .modeling_whisper import MobilintWhisperForConditionalGeneration
    from .processing_whisper import MobilintWhisperFeatureExtractor


def __getattr__(name: str):
    import importlib

    if name == "MobilintWhisperConfig":
        module = importlib.import_module(".configuration_whisper", __package__)
        return module.MobilintWhisperConfig

    if name == "MobilintWhisperForConditionalGeneration":
        module = importlib.import_module(".modeling_whisper", __package__)
        return module.MobilintWhisperForConditionalGeneration

    if name == "MobilintWhisperFeatureExtractor":
        module = importlib.import_module(".processing_whisper", __package__)
        return module.MobilintWhisperFeatureExtractor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MobilintWhisperConfig",
    "MobilintWhisperFeatureExtractor",
    "MobilintWhisperForConditionalGeneration",
]
