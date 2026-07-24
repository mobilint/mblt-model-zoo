"""Tests for YOLO26 ADE20K semantic segmentation support."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mblt_model_zoo.vision import list_models
from mblt_model_zoo.vision.utils.datasets import CustomADE20K, get_ade20k_loader, get_ade20k_palette
from mblt_model_zoo.vision.utils.evaluation import SemanticMetricAccumulator, calculate_semantic_metrics
from mblt_model_zoo.vision.utils.postprocess import SemanticSegPost
from mblt_model_zoo.vision.utils.preprocess import build_preprocess
from mblt_model_zoo.vision.utils.results import Results
from mblt_model_zoo.vision.wrapper import resolve_model_config


def test_yolo26_ade20k_configs_resolve_onnx_artifacts() -> None:
    """Expose every ADE20K model and its matching Hub ONNX artifact."""

    assert list_models("semantic_segmentation")["semantic_segmentation"] == [
        "YOLO26lSemADE20K",
        "YOLO26mSemADE20K",
        "YOLO26nSemADE20K",
        "YOLO26sSemADE20K",
        "YOLO26xSemADE20K",
    ]
    for size in "nsmlx":
        config = resolve_model_config(f"yolo26{size}-sem-ade20k")
        assert config["file_cfg"]["repo_id"] == f"mobilint/YOLO26{size}-sem-ade20k"
        assert config["file_cfg"]["onnx_filename"] == f"yolo26{size}-sem-ade20k.onnx"
        assert config["pre_cfg"]["LetterBox"]["img_size"] == [640, 640]
        assert config["post_cfg"] == {"task": "semantic_segmentation", "dataset": "ade20k"}


def test_semantic_postprocess_supports_logits_and_baked_maps() -> None:
    """Convert logits or baked maps to input-sized integer class maps."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
    )
    logits = torch.zeros((1, 150, 2, 2))
    logits[:, 7] = 1.0
    result = post(logits)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 4, 4)
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.full((1, 4, 4), 7))

    baked = post(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    assert isinstance(baked, torch.Tensor)
    assert baked.shape == (1, 4, 4)
    assert set(baked.unique().tolist()) == {1, 2, 3, 4}

    with pytest.raises(ValueError, match=r"expects \[B, 150, H, W\]"):
        post(torch.zeros((1, 19, 4, 4)))
    with pytest.raises(ValueError, match=r"must be in \[0, 149\]"):
        post(torch.full((1, 4, 4), 150))


def test_semantic_postprocess_restores_letterbox_padding() -> None:
    """Crop padding before nearest-restoring a semantic map."""

    post = SemanticSegPost(
        {"LetterBox": {"img_size": [8, 8]}},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
    )
    output = torch.zeros((1, 8, 8))
    output[:, 2:6, :] = 5
    restored = post(output, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    assert torch.equal(restored, torch.full((2, 4), 5))


def test_ade20k_loader_maps_labels_and_letterboxes_masks(tmp_path: Path) -> None:
    """Pair flat ADE20K files and apply image-equivalent letterbox geometry."""

    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    annotation = np.array([[0, 1, 2, 150], [1, 2, 3, 4]], dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
    assert cv2.imwrite(str(annotation_dir / "sample.png"), annotation)
    preprocessor = build_preprocess(
        {
            "Reader": {"style": "numpy"},
            "LetterBox": {"img_size": [4, 4]},
            "SetOrder": {"shape": "HWC"},
            "Normalize": {"style": "cv"},
        }
    )
    loader = get_ade20k_loader(
        CustomADE20K(str(tmp_path)),
        batch_size=1,
        preprocess_fn=preprocessor.with_metadata,
        image_size=(4, 4),
    )
    inputs, targets, shapes, ratio_pads, stems = next(iter(loader))
    assert inputs.shape == (1, 4, 4, 3)
    assert targets.shape == (1, 4, 4)
    assert np.all(targets[0, 0] == 255)
    assert np.array_equal(targets[0, 1], np.array([255, 0, 1, 149], dtype=np.uint8))
    assert np.array_equal(targets[0, 2], np.array([0, 1, 2, 3], dtype=np.uint8))
    assert np.all(targets[0, 3] == 255)
    assert shapes == [(2, 4)]
    assert ratio_pads == [((1.0, 1.0), (0, 1))]
    assert stems == ("sample",)


def test_ade20k_dataset_rejects_unpaired_files(tmp_path: Path) -> None:
    """Reject partial ADE20K image/annotation layouts."""

    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    assert cv2.imwrite(str(tmp_path / "images" / "sample.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="mismatch"):
        CustomADE20K(str(tmp_path))


def test_semantic_metrics_ignore_void_pixels_and_pool_counts() -> None:
    """Compute mIoU and pixel accuracy while excluding label 255."""

    target = np.array([[0, 0, 1], [1, 255, 2]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1], [1, 0, 2]], dtype=np.uint8)
    result = calculate_semantic_metrics(prediction, target, nc=3)
    assert result.miou == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert result.pixel_accuracy == pytest.approx(4 / 5)

    accumulator = SemanticMetricAccumulator(nc=3)
    accumulator.update(prediction, target)
    accumulator.update(np.array([[2]], dtype=np.uint8), np.array([[2]], dtype=np.uint8))
    pooled = accumulator.result()
    assert pooled.pixel_accuracy == pytest.approx(5 / 6)


def test_semantic_results_plot_restores_original_shape(tmp_path: Path) -> None:
    """Save a semantic overlay at the source image dimensions."""

    source_path = tmp_path / "source.jpg"
    save_path = tmp_path / "semantic.jpg"
    assert cv2.imwrite(str(source_path), np.full((4, 8, 3), 255, dtype=np.uint8))
    class_map = torch.zeros((1, 8, 8), dtype=torch.int64)
    class_map[:, 2:6, :] = 7
    result = Results(
        {"LetterBox": {"img_size": [8, 8]}},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
        class_map,
    )
    plotted = result.plot(str(source_path), str(save_path))
    assert plotted is not None
    assert plotted.shape == (4, 8, 3)
    expected = cv2.addWeighted(
        np.full((4, 8, 3), 255, dtype=np.uint8),
        0.3,
        np.broadcast_to(np.array([0, 237, 204], dtype=np.uint8), (4, 8, 3)),
        0.7,
        0,
    )
    assert np.array_equal(plotted, expected)
    assert save_path.is_file()


def test_semantic_results_plot_distinguishes_person_and_bus() -> None:
    """Use visibly different palette colors for person and bus classes."""

    assert get_ade20k_palette(12) == (255, 255, 0)
    assert get_ade20k_palette(80) == (255, 42, 4)
    class_map = torch.tensor([[12, 80]], dtype=torch.int64)
    result = Results(
        {},
        {"task": "semantic_segmentation", "dataset": "ade20k"},
        class_map,
    )
    plotted = result.plot(np.zeros((1, 2, 3), dtype=np.uint8))
    expected_overlay = np.array([[[255, 255, 0], [255, 42, 4]]], dtype=np.uint8)
    expected = cv2.addWeighted(np.zeros_like(expected_overlay), 0.3, expected_overlay, 0.7, 0)

    assert np.array_equal(plotted, expected)
    assert not np.array_equal(plotted[0, 0], plotted[0, 1])
