"""Tests for vision plotting helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import mblt_model_zoo.vision.utils.results as results_module
from mblt_model_zoo.vision.utils.datasets import get_dotav1_palette
from mblt_model_zoo.vision.utils.results import Results


def test_image_classification_plot_saves_without_gui_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Save classification results without requiring OpenCV GUI support."""

    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "result.jpg"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(source_path), image)

    def _raise_destroy_all_windows() -> None:
        raise cv2.error("cvDestroyAllWindows is unavailable")

    monkeypatch.setattr(cv2, "destroyAllWindows", _raise_destroy_all_windows)

    output = torch.zeros(1000)
    output[980] = 0.9
    result = Results({}, {"task": "image_classification"}, output)

    plotted = result.plot(str(source_path), str(save_path), topk=1)

    assert plotted is not None
    assert save_path.is_file()


def test_instance_segmentation_plot_supports_nonzero_coco_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plot segmentation results when detections use regular COCO class ids."""

    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "segmentation.jpg"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(source_path), image)

    monkeypatch.setattr(results_module, "scale_boxes", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(results_module, "scale_masks", lambda mask, img0_shape: mask)
    monkeypatch.setattr(results_module, "crop_mask", lambda mask, boxes: mask)

    box_cls = torch.tensor([[4.0, 6.0, 20.0, 24.0, 0.9, 45.0]], dtype=torch.float32)
    mask = torch.ones((1, 32, 32), dtype=torch.float32)
    result = Results(
        {"LetterBox": {"img_size": (32, 32)}},
        {"task": "instance_segmentation"},
        [[box_cls, mask]],
    )

    plotted = result.plot(str(source_path), str(save_path))

    assert plotted is not None
    assert save_path.is_file()


def test_obb_plot_uses_dotav1_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot DOTAv1 boxes without consulting the COCO palette."""

    def _reject_coco_palette(label_idx: int) -> tuple[int, int, int]:
        raise AssertionError(f"Unexpected COCO palette lookup for DOTAv1 class {label_idx}.")

    monkeypatch.setattr(results_module, "get_coco_det_palette", _reject_coco_palette)
    box_cls = torch.tensor([[16.0, 16.0, 10.0, 10.0, 0.9, 2.0, 0.0]], dtype=torch.float32)
    result = Results(
        {"LetterBox": {"img_size": (32, 32)}},
        {"task": "obb"},
        [box_cls],
    )

    plotted = result.plot(np.zeros((32, 32, 3), dtype=np.uint8))

    assert plotted is not None
    assert np.any(np.all(plotted == get_dotav1_palette(2), axis=2))
