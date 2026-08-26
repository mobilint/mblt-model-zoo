"""Deprecated compatibility facade for :mod:`mblt_vision`."""

from __future__ import annotations

import sys
from typing import Any

import mblt_vision as _standalone_vision

MBLT_Engine = _standalone_vision.MBLT_Engine
list_models = _standalone_vision.list_models
list_tasks = _standalone_vision.list_tasks


def _register_task_modules() -> None:
    """Expose every standalone Vision task as a deep-importable submodule here
    (``from mblt_model_zoo.vision.<task> import X``) without a physical stub
    package per task, so a new task added to mblt_vision needs no Model Zoo edit.
    """

    for task in _standalone_vision.list_tasks():
        task_module = getattr(_standalone_vision, task)
        globals()[task] = task_module
        sys.modules[f"{__name__}.{task}"] = task_module


_register_task_modules()

# ``obb`` is canonical; retain the historical package alias.
obb = _standalone_vision.obb
oriented_bounding_boxes = obb
sys.modules[f"{__name__}.oriented_bounding_boxes"] = obb

__all__ = [*_standalone_vision.__all__, "oriented_bounding_boxes"]


def __getattr__(name: str) -> Any:
    """Forward standalone Vision exports through the historical package path."""

    return getattr(_standalone_vision, name)


def __dir__() -> list[str]:
    """Return compatibility and standalone Vision attributes."""

    return sorted(set(globals()) | set(__all__))
