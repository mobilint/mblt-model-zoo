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

from ._api import list_models, list_tasks
from .face_detection import *
from .image_classification import *
from .instance_segmentation import *
from .object_detection import *
from .pose_estimation import *
