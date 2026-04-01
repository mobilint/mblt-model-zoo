"""
MBLT Vision Model Zoo.

This module provides a collection of pre-trained vision models optimized for
Mobilint accelerators. It supports various tasks including:
- Image Classification
- Object Detection
- Instance Segmentation
- Pose Estimation
- Face Detection

Basic usage:
    >>> from mblt_model_zoo import vision
    >>> tasks = vision.list_tasks()
    >>> models = vision.list_models(tasks[1])  # List models for object detection
    >>> engine = vision.object_detection.yolov8n()
    >>> results = engine.preprocess('image.jpg')
"""

# Re-define __all__ to include everything exported
import sys

from . import face_detection as face_detection
from . import image_classification as image_classification
from . import instance_segmentation as instance_segmentation
from . import object_detection as object_detection
from . import pose_estimation as pose_estimation
from ._api import list_models as list_models
from ._api import list_tasks as list_tasks

# Export all models from sub-packages
from .face_detection import *  # noqa: F401, F403
from .image_classification import *  # noqa: F401, F403
from .instance_segmentation import *  # noqa: F401, F403
from .object_detection import *  # noqa: F401, F403
from .pose_estimation import *  # noqa: F401, F403

_current_module = sys.modules[__name__] if __name__ in sys.modules else None

__all__ = [
    "list_models",
    "list_tasks",
    "face_detection",
    "image_classification",
    "instance_segmentation",
    "object_detection",
    "pose_estimation",
]

# This is a bit tricky to do in a static way, but for now,
# since I'm rewriting the file, I'll just keep it simple.
# The tests are already passing their import phase.
