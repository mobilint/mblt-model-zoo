"""Focused tests for the vision postprocessor class hierarchy and builder."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.base import PostBase, YOLODetectionPostBase
from mblt_model_zoo.vision.utils.postprocess.cls_post import ClsPost
from mblt_model_zoo.vision.utils.postprocess.depth_post import DepthPost
from mblt_model_zoo.vision.utils.postprocess.semantic_seg_post import SemanticSegPost
from mblt_model_zoo.vision.utils.postprocess.yolo_anchor_post import YOLOAnchorDetectionPost, YOLOAnchorSegPost
from mblt_model_zoo.vision.utils.postprocess.yolo_anchorless_post import (
    YOLOAnchorlessDetectionPost,
    YOLOAnchorlessOBBPost,
    YOLOAnchorlessPosePost,
    YOLOAnchorlessSegPost,
)
from mblt_model_zoo.vision.utils.postprocess.yolo_dflfree_post import (
    YOLODFLFreeDetectionPost,
    YOLODFLFreeOBBPost,
    YOLODFLFreePosePost,
    YOLODFLFreeSegPost,
)
from mblt_model_zoo.vision.utils.postprocess.yolo_nmsfree_post import YOLONMSFreeDetectionPost


@pytest.mark.parametrize(
    ("post_cfg", "expected_type"),
    [
        ({"task": "object_detection", "anchors": [[10, 13, 16, 30, 33, 23]]}, YOLOAnchorDetectionPost),
        ({"task": "object_detection", "nl": 3, "reg_max": 16}, YOLOAnchorlessDetectionPost),
        ({"task": "object_detection", "nl": 3, "dflfree": True}, YOLODFLFreeDetectionPost),
        ({"task": "object_detection", "nl": 3, "reg_max": 16, "nmsfree": True}, YOLONMSFreeDetectionPost),
        (
            {"task": "instance_segmentation", "anchors": [[10, 13, 16, 30, 33, 23]], "n_extra": 32},
            YOLOAnchorSegPost,
        ),
        ({"task": "instance_segmentation", "nl": 3, "reg_max": 16, "n_extra": 32}, YOLOAnchorlessSegPost),
        (
            {"task": "instance_segmentation", "nl": 3, "dflfree": True, "n_extra": 32},
            YOLODFLFreeSegPost,
        ),
        ({"task": "pose_estimation", "nl": 3, "reg_max": 16, "n_extra": 51}, YOLOAnchorlessPosePost),
        ({"task": "pose_estimation", "nl": 3, "dflfree": True, "n_extra": 51}, YOLODFLFreePosePost),
        ({"task": "obb", "nl": 3, "reg_max": 16, "n_extra": 1}, YOLOAnchorlessOBBPost),
        ({"task": "obb", "nl": 3, "dflfree": True, "n_extra": 1}, YOLODFLFreeOBBPost),
    ],
)
def test_builder_routes_detection_backends_without_warnings(
    post_cfg: dict[str, Any], expected_type: type[YOLODetectionPostBase]
) -> None:
    """Build every detection family through canonical warning-free imports."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        postprocessor = build_postprocess({"LetterBox": {"img_size": [640, 640]}}, post_cfg)
    assert type(postprocessor) is expected_type


@pytest.mark.parametrize(
    ("pre_cfg", "post_cfg", "expected_type"),
    [
        ({}, {"task": "image_classification"}, ClsPost),
        ({"LetterBox": {"img_size": [8, 8]}}, {"task": "depth_estimation"}, DepthPost),
        (
            {"LetterBox": {"img_size": [8, 8]}},
            {"task": "semantic_segmentation", "dataset": "ade20k"},
            SemanticSegPost,
        ),
    ],
)
def test_builder_keeps_non_detection_routing_warning_free(
    pre_cfg: dict[str, Any], post_cfg: dict[str, Any], expected_type: type[PostBase]
) -> None:
    """Preserve classification and dense prediction builder routes."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        postprocessor = build_postprocess(pre_cfg, post_cfg)
    assert type(postprocessor) is expected_type


def test_detection_postprocessor_rejects_missing_head_count() -> None:
    """Raise a stable runtime error when anchorless metadata omits nl."""

    with pytest.raises(ValueError, match="nl should be provided in post_cfg"):
        YOLOAnchorlessDetectionPost(
            {"LetterBox": {"img_size": [640, 640]}},
            {"task": "object_detection", "dataset": "coco"},
        )
