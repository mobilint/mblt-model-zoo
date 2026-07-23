"""NYU Depth V2 evaluation for monocular depth-estimation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from ..datasets import CustomNYUDepth, get_nyu_depth_loader
from ..preprocess import build_preprocess

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


@dataclass(frozen=True)
class NYUDepthResult:
    """Median-aligned NYU Depth V2 metrics."""

    delta1: float
    abs_rel: float
    rmse: float


class NYUDepthMetricAccumulator:
    """Accumulate median-aligned metrics over every valid NYU depth pixel."""

    MIN_DEPTH = 0.001
    MAX_DEPTH = 100.0

    def __init__(self) -> None:
        """Initialize zero-valued pixel sums."""

        self.delta1_sum = 0.0
        self.abs_rel_sum = 0.0
        self.squared_error_sum = 0.0
        self.valid_pixel_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Median-align one prediction and add its valid-pixel statistics."""

        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if prediction.shape != target.shape:
            raise ValueError(
                f"NYU Depth prediction and target shapes must match, got {prediction.shape} and {target.shape}."
            )
        valid = np.isfinite(prediction) & np.isfinite(target) & (target > self.MIN_DEPTH) & (target < self.MAX_DEPTH)
        valid_pixel_count = int(valid.sum())
        if valid_pixel_count == 0:
            raise ValueError("NYU Depth sample has no valid pixels in the (0.001, 100.0) range.")

        predicted, actual = prediction[valid], target[valid]
        median_index = (valid_pixel_count - 1) // 2
        median_prediction = np.partition(np.maximum(predicted, self.MIN_DEPTH), median_index)[median_index]
        median_target = np.partition(actual, median_index)[median_index]
        aligned = predicted * (median_target / median_prediction)
        aligned = np.clip(aligned, self.MIN_DEPTH, self.MAX_DEPTH)
        ratio = np.maximum(actual / aligned, aligned / actual)
        self.delta1_sum += float(np.sum(ratio < 1.25))
        self.abs_rel_sum += float(np.sum(np.abs(actual - aligned) / actual))
        self.squared_error_sum += float(np.sum((actual - aligned) ** 2))
        self.valid_pixel_count += valid_pixel_count

    def result(self) -> NYUDepthResult:
        """Return metrics pooled over all accumulated valid pixels."""

        if self.valid_pixel_count == 0:
            raise ValueError("NYU Depth evaluation received no valid pixels.")
        return NYUDepthResult(
            delta1=self.delta1_sum / self.valid_pixel_count,
            abs_rel=self.abs_rel_sum / self.valid_pixel_count,
            rmse=float(np.sqrt(self.squared_error_sum / self.valid_pixel_count)),
        )


def calculate_nyu_depth_metrics(prediction: np.ndarray, target: np.ndarray) -> NYUDepthResult:
    """Calculate official pooled metrics for one median-aligned NYU sample."""

    accumulator = NYUDepthMetricAccumulator()
    accumulator.update(prediction, target)
    return accumulator.result()


def eval_nyu_depth(model: MBLT_Engine, data_path: str, batch_size: int) -> NYUDepthResult:
    """Evaluate a depth model on paired NYU validation images and depth maps."""

    dataset = CustomNYUDepth(data_path)
    letterbox_cfg = model.pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, dict) or "img_size" not in letterbox_cfg:
        raise ValueError("NYU Depth validation requires a LetterBox img_size in the model preprocessing config.")
    image_size = letterbox_cfg["img_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("NYU Depth validation img_size must be a two-item [height, width] list.")

    validation_pre_cfg = {name: config for name, config in model.pre_cfg.items() if name != "LetterBox"}
    validation_preprocessor = build_preprocess(validation_pre_cfg)
    loader = get_nyu_depth_loader(
        dataset,
        batch_size,
        validation_preprocessor,
        image_size=(int(image_size[0]), int(image_size[1])),
    )
    accumulator = NYUDepthMetricAccumulator()
    for inputs, targets, _, _, _ in tqdm(loader, desc="Evaluating NYU Depth"):
        output = model(inputs)
        result = model.postprocess(output)
        depth = result.depth
        if depth is None:
            raise ValueError("Depth postprocessor returned no depth maps.")
        if isinstance(depth, list):
            maps = depth
        elif len(targets) == 1 and depth.ndim == 2:
            maps = [depth]
        else:
            maps = [depth[index] for index in range(len(targets))]
        if len(maps) != len(targets):
            raise ValueError(f"Depth postprocessor returned {len(maps)} maps for {len(targets)} targets.")
        for prediction, target in zip(maps, targets):
            array = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
            accumulator.update(array, target)
    return accumulator.result()
