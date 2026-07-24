"""Focused tests for the vision postprocessor class hierarchy and builder."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from mblt_model_zoo.vision.utils.postprocess import base as base_module
from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess import yolo_anchor_post as anchor_module
from mblt_model_zoo.vision.utils.postprocess import yolo_anchorless_post as anchorless_module
from mblt_model_zoo.vision.utils.postprocess import yolo_dflfree_post as dflfree_module
from mblt_model_zoo.vision.utils.postprocess import yolo_nmsfree_post as nmsfree_module
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


def test_detection_task_classes_inherit_from_canonical_detectors() -> None:
    """Keep each task specialization attached to its detection implementation."""

    for detector_type in (YOLOAnchorDetectionPost, YOLOAnchorlessDetectionPost, YOLODFLFreeDetectionPost):
        assert issubclass(detector_type, YOLODetectionPostBase)
    assert issubclass(YOLOAnchorSegPost, YOLOAnchorDetectionPost)
    assert issubclass(YOLOAnchorlessSegPost, YOLOAnchorlessDetectionPost)
    assert issubclass(YOLOAnchorlessPosePost, YOLOAnchorlessDetectionPost)
    assert issubclass(YOLOAnchorlessOBBPost, YOLOAnchorlessDetectionPost)
    assert issubclass(YOLODFLFreeSegPost, YOLODFLFreeDetectionPost)
    assert issubclass(YOLODFLFreePosePost, YOLODFLFreeDetectionPost)
    assert issubclass(YOLODFLFreeOBBPost, YOLODFLFreeDetectionPost)
    assert issubclass(YOLONMSFreeDetectionPost, YOLOAnchorlessDetectionPost)


def test_dense_task_postprocessors_are_independent_postbase_descendants() -> None:
    """Avoid inheritance between unrelated dense prediction tasks."""

    for postprocessor_type in (ClsPost, DepthPost, SemanticSegPost):
        assert issubclass(postprocessor_type, PostBase)
    assert not issubclass(ClsPost, DepthPost)
    assert not issubclass(ClsPost, SemanticSegPost)
    assert not issubclass(DepthPost, SemanticSegPost)
    assert not issubclass(SemanticSegPost, DepthPost)


def test_legacy_detector_class_names_are_not_exposed() -> None:
    """Keep internal detector modules limited to canonical class names."""

    assert not hasattr(base_module, "YOLOPostBase")
    assert not hasattr(anchor_module, "YOLOAnchorPost")
    assert not hasattr(anchorless_module, "YOLOAnchorlessPost")
    assert not hasattr(dflfree_module, "YOLODFLFreePost")
    assert not hasattr(nmsfree_module, "YOLONMSFreePost")


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
