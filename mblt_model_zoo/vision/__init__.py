"""MBLT vision task exports and discovery helpers.

In ``2.0.0``, ``mblt_model_zoo.vision`` no longer re-exports every legacy model
class at the package top level. Use task subpackages such as
``mblt_model_zoo.vision.image_classification`` for compatibility imports, or
load models through ``MBLT_Engine`` and ``list_models()``.
"""

from __future__ import annotations

from . import face_detection as face_detection
from . import image_classification as image_classification
from . import instance_segmentation as instance_segmentation
from . import object_detection as object_detection
from . import pose_estimation as pose_estimation
from ._api import list_models as list_models
from ._api import list_tasks as list_tasks
from .wrapper import MBLT_Engine as MBLT_Engine

__all__: list[str] = [
    "MBLT_Engine",
    "list_models",
    "list_tasks",
    "face_detection",
    "image_classification",
    "instance_segmentation",
    "object_detection",
    "pose_estimation",
]
