from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen3_asr import (
        MobilintQwen3ASRAudioEncoderConfig,
        MobilintQwen3ASRConfig,
        MobilintQwen3ASRTextConfig,
        MobilintQwen3ASRThinkerConfig,
    )
    from .modeling_qwen3_asr import (
        MobilintQwen3ASRForConditionalGeneration,
        MobilintQwen3ASRThinkerForConditionalGeneration,
    )
    from .processing_qwen3_asr import MobilintQwen3ASRProcessor


def __getattr__(name: str):
    import importlib

    if name in {
        "MobilintQwen3ASRAudioEncoderConfig",
        "MobilintQwen3ASRTextConfig",
        "MobilintQwen3ASRThinkerConfig",
        "MobilintQwen3ASRConfig",
    }:
        module = importlib.import_module(".configuration_qwen3_asr", __package__)
        return getattr(module, name)

    if name in {
        "MobilintQwen3ASRForConditionalGeneration",
        "MobilintQwen3ASRThinkerForConditionalGeneration",
    }:
        module = importlib.import_module(".modeling_qwen3_asr", __package__)
        return getattr(module, name)

    if name == "MobilintQwen3ASRProcessor":
        module = importlib.import_module(".processing_qwen3_asr", __package__)
        return module.MobilintQwen3ASRProcessor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MobilintQwen3ASRAudioEncoderConfig",
    "MobilintQwen3ASRTextConfig",
    "MobilintQwen3ASRThinkerConfig",
    "MobilintQwen3ASRConfig",
    "MobilintQwen3ASRForConditionalGeneration",
    "MobilintQwen3ASRThinkerForConditionalGeneration",
    "MobilintQwen3ASRProcessor",
]
