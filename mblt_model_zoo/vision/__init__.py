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
    >>> results = engine.preprocess("image.jpg")
"""

from . import face_detection as face_detection
from . import image_classification as image_classification
from . import instance_segmentation as instance_segmentation
from . import object_detection as object_detection
from . import pose_estimation as pose_estimation
from ._api import list_models as list_models
from ._api import list_tasks as list_tasks

__all__ = [
    "list_models",
    "list_tasks",
    "face_detection",
    "image_classification",
    "instance_segmentation",
    "object_detection",
    "pose_estimation",
]
