"""Tests for the public vision discovery helpers."""

from __future__ import annotations

import importlib

from mblt_model_zoo import vision
from mblt_model_zoo.vision import list_models, list_tasks


def test_list_tasks_includes_obb() -> None:
    """Advertise the OBB task key used by model configs and validation."""

    assert list_tasks() == [
        "image_classification",
        "depth_estimation",
        "object_detection",
        "instance_segmentation",
        "semantic_segmentation",
        "obb",
        "pose_estimation",
        "face_detection",
    ]


def test_list_models_accepts_obb() -> None:
    """Discover oriented-bounding-box models through the OBB task key."""

    assert list_models("obb")["obb"]


def test_oriented_bounding_boxes_remains_an_obb_alias() -> None:
    """Keep legacy OBB package and discovery names without making them canonical."""

    canonical_models = list_models("obb")["obb"]
    alias_models = list_models("oriented_bounding_boxes")["oriented_bounding_boxes"]
    compatibility_module = importlib.import_module("mblt_model_zoo.vision.oriented_bounding_boxes")

    assert "oriented_bounding_boxes" not in list_tasks()
    assert vision.oriented_bounding_boxes is vision.obb
    assert alias_models == canonical_models
    assert compatibility_module.YOLO26mOBB is vision.obb.YOLO26mOBB
