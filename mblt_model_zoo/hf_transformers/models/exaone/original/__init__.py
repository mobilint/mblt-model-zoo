from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_exaone import *

def __getattr__(name: str):
    import importlib

    if name == "ExaoneConfig":
        module = importlib.import_module(".configuration_exaone", __package__)
        return module.ExaoneConfig

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ExaoneConfig"]