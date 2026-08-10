"""Unit tests for YOLO26 depth-estimation support."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mblt_model_zoo.vision.utils.datasets.dataloader import CustomNYUDepth, get_nyu_depth_loader
from mblt_model_zoo.vision.utils.evaluation import NYUDepthMetricAccumulator, calculate_nyu_depth_metrics
from mblt_model_zoo.vision.utils.postprocess import DepthPost
from mblt_model_zoo.vision.utils.results import Results
from mblt_model_zoo.vision.wrapper import resolve_model_config


@pytest.mark.parametrize(
    ("model_name", "task"),
    [
        ("yolo26s", "object_detection"),
        ("yolo26s-seg", "instance_segmentation"),
        ("yolo26s-pose", "pose_estimation"),
        ("yolo26s-obb", "obb"),
    ],
)
def test_other_yolo_validation_tasks_keep_letterbox(model_name: str, task: str) -> None:
    """Keep aspect-preserving letterbox preprocessing for non-depth YOLO tasks."""

    config = resolve_model_config(model_name)
    assert "LetterBox" in config["pre_cfg"]
    assert "Resize" not in config["pre_cfg"]
    assert config["post_cfg"]["task"] == task


def test_depth_post_restores_letterbox_padding() -> None:
    """Crop letterbox padding before bilinearly restoring an original image shape."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    output = torch.zeros((1, 1, 8, 8))
    output[:, :, 2:6, :] = 2.0
    restored = post(output, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    assert torch.allclose(restored, torch.full((2, 4), 2.0))
    with pytest.raises(ValueError, match=r"expects \[B, 1, H, W\] or \[B, H, W, 1\]"):
        post(torch.zeros((1, 2, 8, 8)))


def test_depth_post_normalizes_quarter_resolution_mxq_before_restoring() -> None:
    """Upsample the MXQ depth layout before undoing letterbox padding."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    mxq_depth = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    expected = torch.nn.functional.interpolate(
        mxq_depth[:, None], scale_factor=4.0, mode="bilinear", align_corners=False
    )[:, 0]

    normalized = post(mxq_depth)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, expected)

    restored = post(mxq_depth, img0_shape=(2, 4), ratio_pad=((2.0, 2.0), (0.0, 2.0)))
    assert isinstance(restored, torch.Tensor)
    assert restored.shape == (2, 4)
    expected_restored = torch.nn.functional.interpolate(
        expected[0, 2:6][None, None], size=(2, 4), mode="bilinear", align_corners=False
    )[0, 0]
    assert torch.equal(restored, expected_restored)


def test_depth_post_keeps_full_resolution_onnx_output_and_rejects_other_scales() -> None:
    """Keep ONNX depth unchanged while rejecting unsupported dense output scales."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    full_resolution = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    normalized = post(full_resolution)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, full_resolution[:, 0])

    with pytest.raises(ValueError, match="spatial shape must be"):
        post(torch.zeros((1, 3, 3)))


def test_depth_post_normalizes_full_resolution_channel_last_mxq_output() -> None:
    """Normalize batched and single-image baked MXQ resize outputs without resizing again."""

    post = DepthPost({"LetterBox": {"img_size": [8, 8]}}, {})
    full_resolution = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8, 1)
    normalized = post(full_resolution)
    assert isinstance(normalized, torch.Tensor)
    assert torch.equal(normalized, full_resolution[..., 0])

    single_image = full_resolution[0]
    normalized_single_image = post(single_image)
    assert isinstance(normalized_single_image, torch.Tensor)
    assert torch.equal(normalized_single_image, full_resolution[..., 0])


def test_depth_metrics_median_align_and_ignore_invalid_targets() -> None:
    """Apply per-image median scale alignment and exclude invalid NYU targets."""

    target = np.array([[1.0, 2.0], [4.0, np.nan]], dtype=np.float32)
    prediction = np.array([[2.0, 4.0], [8.0, np.nan]], dtype=np.float32)
    metrics = calculate_nyu_depth_metrics(prediction, target)
    assert metrics.delta1 == pytest.approx(1.0)
    assert metrics.abs_rel == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.primary_score == metrics.delta1
    assert metrics.secondary_score == metrics.abs_rel


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -np.inf])
def test_depth_metrics_reject_non_finite_predictions_at_valid_targets(invalid_prediction: float) -> None:
    """Do not remove invalid model outputs from the metric denominator."""

    prediction = np.array([[1.0, invalid_prediction]], dtype=np.float32)
    target = np.array([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"1 non-finite value\(s\) at valid target pixels"):
        calculate_nyu_depth_metrics(prediction, target)


def test_depth_metrics_pool_valid_pixels_across_images() -> None:
    """Weight accumulated metrics by valid pixels and clamp aligned predictions."""

    accumulator = NYUDepthMetricAccumulator()
    accumulator.update(np.array([[1.0, 1.0, 1.0]]), np.array([[1.0, 1.0, 10.0]]))
    accumulator.update(np.array([[2.0]]), np.array([[4.0]]))
    metrics = accumulator.result()
    assert metrics.delta1 == pytest.approx(0.75)
    assert metrics.abs_rel == pytest.approx(0.225)
    assert metrics.rmse == pytest.approx(4.5)

    clamped = calculate_nyu_depth_metrics(
        np.array([[-1.0, 2.0, 1000.0]]),
        np.array([[1.0, 2.0, 4.0]]),
    )
    assert clamped.delta1 == pytest.approx(1 / 3)

    even_pixel_median = calculate_nyu_depth_metrics(
        np.array([[1.0, 4.0]]),
        np.array([[1.0, 10.0]]),
    )
    assert even_pixel_median.delta1 == pytest.approx(0.5)
    assert even_pixel_median.abs_rel == pytest.approx(0.66)
    assert even_pixel_median.rmse == pytest.approx(1.2)


def test_nyu_depth_validation_loader_stretches_rgb_and_target_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stretch each original array directly to the validation input dimensions."""

    image_dir, depth_dir = tmp_path / "images", tmp_path / "depth"
    image_dir.mkdir()
    depth_dir.mkdir()
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    target = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    cv2.imwrite(str(image_dir / "sample.png"), image)
    np.save(depth_dir / "sample.npy", target)

    dataset = CustomNYUDepth(str(tmp_path))
    source_image, source_target, _ = dataset[0]
    original_resize = cv2.resize
    expected_image = original_resize(source_image, (3, 5), interpolation=cv2.INTER_LINEAR)
    expected_target = original_resize(source_target, (3, 5), interpolation=cv2.INTER_NEAREST)
    resize_calls: list[tuple[tuple[int, ...], tuple[int, int], int]] = []

    def _record_resize(array: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        resize_calls.append((array.shape, size, interpolation))
        return original_resize(array, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", _record_resize)
    loader = get_nyu_depth_loader(
        dataset,
        batch_size=1,
        preprocess_fn=lambda value: value,
        image_size=(5, 3),
    )
    inputs, targets, shapes, ratio_pads, stems = next(iter(loader))
    assert np.array_equal(inputs[0], expected_image)
    assert np.array_equal(targets[0], expected_target)
    assert resize_calls == [
        ((2, 4, 3), (3, 5), cv2.INTER_LINEAR),
        ((2, 4), (3, 5), cv2.INTER_NEAREST),
    ]
    assert shapes == [(5, 3)]
    assert ratio_pads == [None]
    assert stems == ("sample",)


def test_nyu_depth_pairing_and_plotting(tmp_path) -> None:
    """Reject unmatched NYU pairs and save an original-size color overlay."""

    image_dir, depth_dir = tmp_path / "images", tmp_path / "depth"
    image_dir.mkdir()
    depth_dir.mkdir()
    cv2.imwrite(str(image_dir / "sample.jpg"), np.zeros((4, 8, 3), dtype=np.uint8))
    np.save(depth_dir / "sample.npy", np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32))
    dataset = CustomNYUDepth(str(tmp_path))
    _, target, _ = dataset[0]
    assert np.isfinite(target).all()

    result = Results(
        {"LetterBox": {"img_size": [8, 8]}},
        {"task": "depth_estimation"},
        torch.ones((1, 8, 8)),
    )
    plotted = result.plot(np.zeros((4, 8, 3), dtype=np.uint8), str(tmp_path / "depth.jpg"))
    assert plotted is not None
    assert plotted.shape == (4, 8, 3)
    colorized_depth = cv2.applyColorMap(np.zeros((4, 8), dtype=np.uint8), cv2.COLORMAP_JET)
    assert np.array_equal(plotted, cv2.addWeighted(np.zeros_like(colorized_depth), 0.4, colorized_depth, 0.6, 0))
    assert (tmp_path / "depth.jpg").is_file()

    np.save(depth_dir / "extra.npy", np.ones((4, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="mismatch"):
        CustomNYUDepth(str(tmp_path))


def test_depth_plot_maps_near_objects_to_red() -> None:
    """Use disparity contrast to render near, middle, and far depths as red, green, and blue."""

    result = Results(
        {},
        {"task": "depth_estimation"},
        torch.tensor([[1.0, 2.0, 100.0]], dtype=torch.float32),
    )

    plotted = result.plot(np.zeros((1, 3, 3), dtype=np.uint8))
    assert plotted is not None
    expected_overlay = cv2.applyColorMap(np.array([[255, 126, 0]], dtype=np.uint8), cv2.COLORMAP_JET)
    expected = cv2.addWeighted(np.zeros_like(expected_overlay), 0.4, expected_overlay, 0.6, 0)

    assert np.array_equal(plotted, expected)
    assert plotted[0, 0, 2] > plotted[0, 0, 0]
    assert plotted[0, 1, 1] > max(plotted[0, 1, 0], plotted[0, 1, 2])
    assert plotted[0, 2, 0] > plotted[0, 2, 2]


def test_depth_results_preserve_restored_non_square_maps() -> None:
    """Avoid applying inverse letterboxing twice to restored depth maps."""

    source = np.zeros((2, 4, 3), dtype=np.uint8)
    restored_depth = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    letterboxed_result = Results(
        {"LetterBox": {"img_size": [4, 4]}},
        {"task": "depth_estimation"},
        restored_depth,
    )
    reference_result = Results({}, {"task": "depth_estimation"}, restored_depth)

    plotted = letterboxed_result.plot(source)
    reference = reference_result.plot(source)

    assert plotted is not None
    assert reference is not None
    assert np.array_equal(plotted, reference)
