"""Deprecated compatibility exports for standalone Vision utilities."""

import importlib
import sys

from mblt_vision.utils.results import Results
from mblt_vision.utils.types import ListTensorLike, TensorLike

for _name in ("preprocess", "postprocess", "datasets", "evaluation"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"mblt_vision.utils.{_name}")

__all__: list[str] = ["Results", "TensorLike", "ListTensorLike"]
