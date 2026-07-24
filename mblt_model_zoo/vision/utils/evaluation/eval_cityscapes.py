"""Cityscapes evaluation for semantic-segmentation models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .eval_ade20k import SemanticSegmentationResult, eval_semantic_segmentation

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


def eval_cityscapes(model: MBLT_Engine, data_path: str, batch_size: int) -> SemanticSegmentationResult:
    """Evaluate a semantic-segmentation model on Cityscapes validation masks."""

    return eval_semantic_segmentation(model, data_path, batch_size, dataset="cityscapes")
