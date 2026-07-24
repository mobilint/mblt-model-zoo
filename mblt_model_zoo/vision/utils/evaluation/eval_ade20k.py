"""ADE20K evaluation for semantic-segmentation models."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import CustomADE20K, CustomCityscapes, get_ade20k_loader, get_cityscapes_loader

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


class ADE20KResult(NamedTuple):
    """ADE20K metrics ordered from primary to secondary."""

    miou: float
    pixel_accuracy: float

    @property
    def primary_score(self) -> float:
        """Return mean intersection-over-union."""

        return self.miou

    @property
    def secondary_score(self) -> float:
        """Return overall valid-pixel accuracy."""

        return self.pixel_accuracy


class SemanticSegmentationResult(NamedTuple):
    """Generic semantic metrics ordered from primary to secondary."""

    miou: float
    pixel_accuracy: float

    @property
    def primary_score(self) -> float:
        """Return mean intersection-over-union."""

        return self.miou

    @property
    def secondary_score(self) -> float:
        """Return overall valid-pixel accuracy."""

        return self.pixel_accuracy


class SemanticMetricAccumulator:
    """Accumulate an ignore-aware semantic confusion matrix."""

    def __init__(self, nc: int, ignore_label: int = 255) -> None:
        """Initialize an empty confusion matrix.

        Args:
            nc: Number of semantic classes.
            ignore_label: Target label excluded from metrics.
        """

        self.nc = nc
        self.ignore_label = ignore_label
        self.matrix = np.zeros((nc, nc), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one or more predicted and target class maps."""

        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Semantic prediction and target shapes must match, got {prediction.shape} and {target.shape}."
            )
        valid = (
            (target != self.ignore_label)
            & (target >= 0)
            & (target < self.nc)
            & (prediction >= 0)
            & (prediction < self.nc)
        )
        if valid.any():
            histogram = np.bincount(
                self.nc * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
                minlength=self.nc**2,
            )
            self.matrix += histogram.reshape(self.nc, self.nc)

    def result(self) -> SemanticSegmentationResult:
        """Compute mIoU over present classes and overall pixel accuracy."""

        ground_truth = self.matrix.sum(axis=1)
        predicted = self.matrix.sum(axis=0)
        intersection = np.diag(self.matrix)
        union = ground_truth + predicted - intersection
        present = ground_truth > 0
        if not present.any():
            raise ValueError("Semantic evaluation received no valid target pixels.")
        iou = np.divide(
            intersection,
            union,
            out=np.zeros(self.nc, dtype=np.float64),
            where=union > 0,
        )
        total = int(self.matrix.sum())
        return SemanticSegmentationResult(
            miou=float(iou[present].mean()),
            pixel_accuracy=float(intersection.sum() / total),
        )


def calculate_semantic_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    nc: int = 150,
    ignore_label: int = 255,
) -> ADE20KResult:
    """Calculate semantic metrics for one batch of class maps."""

    accumulator = SemanticMetricAccumulator(nc=nc, ignore_label=ignore_label)
    accumulator.update(prediction, target)
    result = accumulator.result()
    return ADE20KResult(result.miou, result.pixel_accuracy)


def _evaluate_semantic_loader(
    model: MBLT_Engine,
    loader: torch.utils.data.DataLoader,
    nc: int,
    description: str = "Evaluating semantic segmentation",
) -> SemanticSegmentationResult:
    """Evaluate input-space semantic maps from an organized validation loader."""

    accumulator = SemanticMetricAccumulator(nc=nc)
    for inputs, targets, _, _, _ in tqdm(loader, desc=description):
        result = model.postprocess(model(inputs))
        semantic_mask = result.semantic_mask
        if semantic_mask is None:
            raise ValueError("Semantic postprocessor returned no class maps.")
        prediction = semantic_mask.detach().cpu().numpy() if isinstance(semantic_mask, torch.Tensor) else semantic_mask
        accumulator.update(np.asarray(prediction), targets)
    return accumulator.result()


def eval_semantic_segmentation(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    dataset: str | None = None,
) -> SemanticSegmentationResult:
    """Evaluate a semantic model with the loader for its configured taxonomy.

    Args:
        model: Initialized semantic-segmentation engine.
        data_path: Organized dataset root.
        batch_size: Number of validation samples per inference batch.
        dataset: Optional taxonomy override. Defaults to ``model.post_cfg.dataset``.

    Returns:
        Generic mIoU and pixel-accuracy result.
    """

    letterbox_cfg = model.pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, dict) or "img_size" not in letterbox_cfg:
        raise ValueError("Semantic validation requires a LetterBox img_size in the model preprocessing config.")
    image_size = letterbox_cfg["img_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("Semantic validation img_size must be a two-item [height, width] list.")

    configured_dataset = model.post_cfg.get("dataset")
    taxonomy = (dataset or configured_dataset or "ade20k").lower()
    image_size_tuple = (int(image_size[0]), int(image_size[1]))
    if taxonomy == "ade20k":
        validation_dataset = CustomADE20K(data_path)
        loader = get_ade20k_loader(
            validation_dataset,
            batch_size,
            model.preprocess_with_metadata,
            image_size=image_size_tuple,
        )
        default_nc = 150
        description = "Evaluating ADE20K"
    elif taxonomy == "cityscapes":
        validation_dataset = CustomCityscapes(data_path)
        loader = get_cityscapes_loader(
            validation_dataset,
            batch_size,
            model.preprocess_with_metadata,
            image_size=image_size_tuple,
        )
        default_nc = 19
        description = "Evaluating Cityscapes"
    else:
        raise ValueError(f"Unsupported semantic validation dataset: {taxonomy!r}.")
    nc = int(getattr(model.postprocessor, "nc", default_nc))
    return _evaluate_semantic_loader(model, loader, nc, description=description)


def eval_ade20k(model: MBLT_Engine, data_path: str, batch_size: int) -> ADE20KResult:
    """Evaluate a semantic-segmentation model on ADE20K validation masks."""

    result = eval_semantic_segmentation(
        model,
        data_path,
        batch_size,
        dataset="ade20k",
    )
    return ADE20KResult(result.miou, result.pixel_accuracy)
