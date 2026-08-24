"""Compatibility exports for the standalone Mobilint NPU backend.

New applications should import from :mod:`mblt_npu`.  This module preserves the
historical Model Zoo import path while the implementation is maintained by
``mblt-npu-python``.
"""

from mblt_npu import (
    BACKEND_CLASSES,
    DEFAULT_TARGET_DEVICE,
    MobilintAriesBackend,
    MobilintBackendAllocError,
    MobilintNPUBackend,
    MobilintRegulusBackend,
    backend_class_for,
)


def _get_transformers_dispatcher(backend: MobilintNPUBackend):
    """Attach Model Zoo's transformer dispatcher without coupling the NPU wheel to transformers."""
    dispatcher = getattr(backend, "_mblt_model_zoo_dispatcher", None)
    if dispatcher is None:
        from ..hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher

        dispatcher = MultiSlotDispatcher(backend)
        backend._mblt_model_zoo_dispatcher = dispatcher
    return dispatcher


# ``dispatcher`` is a Model Zoo transformers-integration concern. Keep the
# runtime wheel independent while preserving the historical backend surface for
# existing Model Zoo callers.
if not hasattr(MobilintNPUBackend, "dispatcher"):
    MobilintNPUBackend.dispatcher = property(_get_transformers_dispatcher)

__all__ = [
    "BACKEND_CLASSES",
    "DEFAULT_TARGET_DEVICE",
    "MobilintBackendAllocError",
    "MobilintAriesBackend",
    "MobilintNPUBackend",
    "MobilintRegulusBackend",
    "backend_class_for",
]
