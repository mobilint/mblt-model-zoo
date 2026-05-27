from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen2_eagle3 import *
    from .modeling_qwen2_eagle3 import *


def __getattr__(name: str):
    import importlib

    if name in {"MobilintEagle3DraftConfig", "MobilintQwen2Eagle3Config"}:
        module = importlib.import_module(".configuration_qwen2_eagle3", __package__)
        return getattr(module, name)

    if name in {
        "MobilintEagle3BaseModel",
        "MobilintEagle3DraftModel",
        "MobilintEagle3FCProjector",
        "MobilintQwen2Eagle3ForCausalLM",
        "MobilintQwen2Eagle3Model",
    }:
        module = importlib.import_module(".modeling_qwen2_eagle3", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MobilintEagle3DraftConfig",
    "MobilintEagle3BaseModel",
    "MobilintEagle3DraftModel",
    "MobilintEagle3FCProjector",
    "MobilintQwen2Eagle3Config",
    "MobilintQwen2Eagle3ForCausalLM",
    "MobilintQwen2Eagle3Model",
]
