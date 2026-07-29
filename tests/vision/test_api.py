"""Tests for the public vision discovery helpers."""

from __future__ import annotations

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
