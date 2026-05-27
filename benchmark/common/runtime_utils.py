"""Shared runtime cleanup helpers for benchmark scripts."""

from __future__ import annotations

import gc
from typing import Any


def is_cuda_device(device: str | None) -> bool:
    """Returns whether a device string targets CUDA.

    Args:
        device: Device string from CLI arguments.

    Returns:
        ``True`` when the device starts with ``cuda``.
    """
    return isinstance(device, str) and device.strip().lower().startswith("cuda")


def cuda_device_index(device: str | None) -> int | None:
    """Parses the CUDA device index from a device string.

    Args:
        device: Device string such as ``cuda`` or ``cuda:0``.

    Returns:
        CUDA index, or ``None`` when unavailable.
    """
    if not is_cuda_device(device):
        return None
    text = (device or "").strip().lower()
    if ":" not in text:
        return 0
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return None


def is_cuda_oom_error(exc: Exception) -> bool:
    """Detects common CUDA out-of-memory error messages.

    Args:
        exc: Exception raised by model loading or execution.

    Returns:
        ``True`` when the message looks like CUDA OOM.
    """
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or ("out of memory" in message and "cuda" in message)
        or "cublas_status_alloc_failed" in message
    )


def clear_cuda_memory(device: str | None = None) -> None:
    """Best-effort CUDA memory cleanup.

    Args:
        device: Optional CUDA device string used before cache cleanup.
    """
    try:
        import torch
    except (ImportError, RuntimeError):
        return
    if not torch.cuda.is_available():
        return
    idx = cuda_device_index(device)
    try:
        if idx is not None:
            torch.cuda.set_device(idx)
    except RuntimeError:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def release_pipeline(pipeline_obj: Any, device: str | None = None) -> None:
    """Disposes a pipeline object and clears CUDA memory when needed.

    Args:
        pipeline_obj: Hugging Face or Mobilint pipeline-like object.
        device: Optional device string for CUDA cleanup.
    """
    if pipeline_obj is not None:
        try:
            model_obj = getattr(pipeline_obj, "model", None)
            if model_obj is not None and hasattr(model_obj, "dispose"):
                model_obj.dispose()
            elif hasattr(pipeline_obj, "dispose"):
                pipeline_obj.dispose()
        except (AttributeError, RuntimeError):
            pass
        del pipeline_obj
    gc.collect()
    if is_cuda_device(device):
        clear_cuda_memory(device)