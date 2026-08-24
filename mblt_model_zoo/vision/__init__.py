"""Deprecated compatibility facade for :mod:`mblt_vision`."""

from __future__ import annotations

import sys
from typing import Any

import mblt_vision as _standalone_vision

MBLT_Engine = _standalone_vision.MBLT_Engine
list_models = _standalone_vision.list_models
list_tasks = _standalone_vision.list_tasks
depth_estimation = _standalone_vision.depth_estimation
face_detection = _standalone_vision.face_detection
image_classification = _standalone_vision.image_classification
instance_segmentation = _standalone_vision.instance_segmentation
object_detection = _standalone_vision.object_detection
obb = _standalone_vision.obb
pose_estimation = _standalone_vision.pose_estimation
semantic_segmentation = _standalone_vision.semantic_segmentation

# ``obb`` is canonical; retain the historical package alias.
oriented_bounding_boxes = obb
sys.modules[f"{__name__}.oriented_bounding_boxes"] = obb

__all__ = [*_standalone_vision.__all__, "oriented_bounding_boxes"]


def __getattr__(name: str) -> Any:
    """Forward standalone Vision exports through the historical package path."""

    return getattr(_standalone_vision, name)


def __dir__() -> list[str]:
    """Return compatibility and standalone Vision attributes."""

    return sorted(set(globals()) | set(__all__))
