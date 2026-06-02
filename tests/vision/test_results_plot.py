"""Tests for vision plotting helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

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
