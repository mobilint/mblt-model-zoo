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
from mblt_model_zoo.vision.utils.types import NestedListTensorLike


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


def test_image_classification_plot_rejects_missing_output() -> None:
    """Raise a runtime validation error when classification output is absent."""

    result = Results({}, {"task": "image_classification"}, torch.zeros(1))
    result.acc = None

    with pytest.raises(ValueError, match="No accuracy output found"):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), topk=1)


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


def test_dotav1_palette_wraps_class_indices() -> None:
    """Match the modulo behavior used by the other visualization palettes."""

    assert get_dotav1_palette(15) == get_dotav1_palette(0)


def test_results_accept_path_and_basename_save_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Accept pathlib inputs and avoid creating an empty parent directory."""

    source_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source_path), np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.chdir(tmp_path)
    result = Results({}, {"task": "image_classification"}, torch.arange(3, dtype=torch.float32))

    result.plot(source_path, Path("result.jpg"), topk=10)

    assert (tmp_path / "result.jpg").is_file()


def test_results_report_failed_image_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise OSError when OpenCV cannot encode or write the requested image."""

    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    result = Results({}, {"task": "image_classification"}, torch.ones(2))

    with pytest.raises(OSError, match="Failed to write"):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), "result.jpg", topk=1)


@pytest.mark.parametrize("topk", [0, -1, 1.5, True])
def test_results_validate_classification_topk(topk: object) -> None:
    """Reject invalid classification Top-K values."""

    result = Results({}, {"task": "image_classification"}, torch.ones(2))
    with pytest.raises((TypeError, ValueError)):
        result.plot(np.zeros((8, 8, 3), dtype=np.uint8), topk=topk)


@pytest.mark.parametrize(
    ("task", "output"),
    [
        ("object_detection", []),
        ("instance_segmentation", []),
        ("instance_segmentation", [[]]),
    ],
)
def test_results_reject_empty_structured_outputs(task: str, output: NestedListTensorLike) -> None:
    """Validate structured result containers before indexing."""

    with pytest.raises(ValueError):
        Results({}, {"task": task}, output)


def test_results_normalize_task_alias_and_semantic_taxonomy_case() -> None:
    """Normalize OBB aliases and semantic palette taxonomy casing."""

    obb = Results({}, {"task": "oriented_bounding_boxes"}, [torch.zeros((0, 7))])
    semantic = Results(
        {},
        {"task": "semantic_segmentation", "dataset": "CityScapes"},
        np.zeros((1, 4, 4), dtype=np.uint8),
    )

    assert obb.task == "obb"
    plotted = semantic.plot(np.zeros((4, 4, 3), dtype=np.uint8))
    assert plotted is not None
    assert plotted.shape == (4, 4, 3)
