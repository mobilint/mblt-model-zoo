"""Tests for the public vision discovery helpers."""

from __future__ import annotations

from mblt_model_zoo.vision import list_models, list_tasks


def test_list_tasks_includes_obb_alias() -> None:
    """Advertise the OBB task key used by model configs and validation."""

    assert "obb" in list_tasks()


def test_list_models_accepts_obb_alias() -> None:
    """Resolve the OBB alias to the oriented-bounding-boxes model package."""

    alias_models = list_models("obb")["obb"]
    canonical_models = list_models("oriented_bounding_boxes")["oriented_bounding_boxes"]

    assert alias_models == canonical_models
