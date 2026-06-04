"""Tests for vision package-level exports."""

from __future__ import annotations

from mblt_model_zoo import vision
from mblt_model_zoo.vision.image_classification import ResNet50 as ImageClassificationResNet50
from mblt_model_zoo.vision.object_detection import YOLO11m as ObjectDetectionYOLO11m


def test_vision_package_keeps_legacy_top_level_model_exports() -> None:
    """Expose legacy model class imports from the vision package top level."""

    assert vision.ResNet50 is ImageClassificationResNet50
    assert vision.YOLO11m is ObjectDetectionYOLO11m
