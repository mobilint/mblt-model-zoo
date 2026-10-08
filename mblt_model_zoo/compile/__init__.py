"""Deprecated compatibility exports for :mod:`mblt_vision.compile`."""

from .._deprecation import warn_deprecated_import
from .vision import compile_vision_model

warn_deprecated_import(__name__, "mblt-vision-python", "mblt_vision.compile")

__all__ = ["compile_vision_model"]
