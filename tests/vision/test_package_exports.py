"""Tests for vision package-level exports."""

from __future__ import annotations

from mblt_model_zoo import vision
from mblt_model_zoo.vision import object_detection
from mblt_model_zoo.vision.image_classification import ResNet50 as ImageClassificationResNet50
from mblt_model_zoo.vision.object_detection import YOLO11m as ObjectDetectionYOLO11m


def test_vision_package_keeps_legacy_top_level_model_exports() -> None:
    """Expose legacy model class imports from the vision package top level."""

    assert vision.ResNet50 is ImageClassificationResNet50
    assert vision.YOLO11m is ObjectDetectionYOLO11m


def test_yolo26_distill_models_are_exported_and_discoverable() -> None:
    """Expose every YOLO26 Distill config through task and legacy APIs."""

    expected = {
        "YOLO26lDistill": "YOLO26l-distill",
        "YOLO26mDistill": "YOLO26m-distill",
        "YOLO26nDistill": "YOLO26n-distill",
        "YOLO26sDistill": "YOLO26s-distill",
        "YOLO26xDistill": "YOLO26x-distill",
    }
    discovered = vision.list_models("object_detection")["object_detection"]

    for class_name, yaml_name in expected.items():
        task_class = getattr(object_detection, class_name)
        assert getattr(vision, class_name) is task_class
        assert task_class._yaml_name == yaml_name
        assert class_name in discovered
