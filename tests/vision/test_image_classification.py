"""Compatibility coverage for image-classification vision exports.

Runtime inference coverage lives in ``test_mxq_classification.py`` and
``test_onnx_classification.py``; this suite keeps the legacy import contract
covered without requiring accelerator hardware.
"""

from __future__ import annotations

from mblt_model_zoo import vision
from mblt_model_zoo.vision.image_classification import ResNet50


def test_image_classification_legacy_export_is_available() -> None:
    """Expose ResNet50 from both the legacy and task-specific import paths."""

    assert vision.ResNet50 is ResNet50
    assert "ResNet50" in vision.list_models("image_classification")["image_classification"]
