from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen3_vl import MobilintQwen3VLConfig
    from .modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration

def __getattr__(name: str):
    import importlib

    if name == "MobilintQwen3VLConfig":
        module = importlib.import_module(".configuration_qwen3_vl", __package__)
        return module.MobilintQwen3VLConfig

    if name == "MobilintQwen3VLForConditionalGeneration":
        module = importlib.import_module(".modeling_qwen3_vl", __package__)
        return module.MobilintQwen3VLForConditionalGeneration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintQwen3VLConfig", "MobilintQwen3VLForConditionalGeneration"]