"""Tests for vision plotting helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import mblt_model_zoo.vision.utils.results as results_module
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
