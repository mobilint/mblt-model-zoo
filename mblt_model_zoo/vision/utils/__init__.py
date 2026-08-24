"""Deprecated compatibility exports for standalone Vision utilities."""

import importlib
import pkgutil
import sys

from mblt_vision.utils.results import Results
from mblt_vision.utils.types import ListTensorLike, TensorLike


def _alias_module_tree(legacy_name: str, standalone_name: str) -> None:
    """Register a standalone module tree under its historical import path."""

    module = importlib.import_module(standalone_name)
    sys.modules[legacy_name] = module
    if hasattr(module, "__path__"):
        for child in pkgutil.iter_modules(module.__path__):
            _alias_module_tree(f"{legacy_name}.{child.name}", f"{standalone_name}.{child.name}")


for _name in ("preprocess", "postprocess", "datasets", "evaluation"):
    _alias_module_tree(f"{__name__}.{_name}", f"mblt_vision.utils.{_name}")

__all__: list[str] = ["Results", "TensorLike", "ListTensorLike"]
