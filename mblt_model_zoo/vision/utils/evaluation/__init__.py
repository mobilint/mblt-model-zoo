"""
Evaluation scripts for various datasets.
"""

from __future__ import annotations

from .eval_ade20k import (
    ADE20KResult,
    SemanticMetricAccumulator,
    SemanticSegmentationResult,
    calculate_semantic_metrics,
    eval_ade20k,
    eval_semantic_segmentation,
)
from .eval_cityscapes import eval_cityscapes
from .eval_coco import eval_coco
from .eval_dota import DOTAResult, eval_dota
from .eval_imagenet import ImageNetResult, eval_imagenet
from .eval_nyu_depth import NYUDepthMetricAccumulator, NYUDepthResult, calculate_nyu_depth_metrics, eval_nyu_depth
from .eval_widerface import WiderFaceResult, eval_widerface

__all__: list[str] = [
    "eval_coco",
    "ADE20KResult",
    "SemanticMetricAccumulator",
    "SemanticSegmentationResult",
    "calculate_semantic_metrics",
    "eval_ade20k",
    "eval_cityscapes",
    "eval_semantic_segmentation",
    "DOTAResult",
    "eval_dota",
    "ImageNetResult",
    "eval_imagenet",
    "NYUDepthResult",
    "NYUDepthMetricAccumulator",
    "calculate_nyu_depth_metrics",
    "eval_nyu_depth",
    "WiderFaceResult",
    "eval_widerface",
]
