from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_bert import *
    from .modeling_bert import *

def __getattr__(name: str):
    import importlib

    if name == "MobilintBertConfig":
        module = importlib.import_module(".configuration_bert", __package__)
        return module.MobilintBertConfig
    
    if name == "MobilintBertForMaskedLM":
        module = importlib.import_module(".modeling_bert", __package__)
        return module.MobilintBertForMaskedLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MobilintBertConfig", "MobilintBertForMaskedLM"]