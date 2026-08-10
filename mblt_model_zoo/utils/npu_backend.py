"""Compatibility exports for the standalone Mobilint NPU backend.

New applications should import from :mod:`mblt_npu`.  This module preserves the
historical Model Zoo import path while the implementation is maintained by
``mblt-npu-python``.
"""

from mblt_npu import (
    BACKEND_CLASSES,
    MobilintAriesBackend,
    MobilintNPUBackend,
    MobilintRegulusBackend,
    backend_class_for,
)

__all__ = [
    "BACKEND_CLASSES",
    "MobilintAriesBackend",
    "MobilintNPUBackend",
    "MobilintRegulusBackend",
    "backend_class_for",
]
