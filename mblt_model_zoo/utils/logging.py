"""Compatibility export for standalone Mobilint NPU logging."""

from mblt_npu import log_model_details

from .._deprecation import warn_deprecated_import

warn_deprecated_import(__name__, "mblt-npu-python", "mblt_npu")

__all__ = ["log_model_details"]
