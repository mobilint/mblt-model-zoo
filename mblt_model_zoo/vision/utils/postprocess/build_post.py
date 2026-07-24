"""
Postprocessing builder.
"""

from __future__ import annotations

from .base import PostBase
from .cls_post import ClsPost
from .depth_post import DepthPost
from .semantic_seg_post import SemanticSegPost
from .yolo_anchor_post import YOLOAnchorDetectionPost, YOLOAnchorSegPost
from .yolo_anchorless_post import (
    YOLOAnchorlessDetectionPost,
    YOLOAnchorlessOBBPost,
    YOLOAnchorlessPosePost,
    YOLOAnchorlessSegPost,
)
from .yolo_dflfree_post import (
    YOLODFLFreeDetectionPost,
    YOLODFLFreeOBBPost,
    YOLODFLFreePosePost,
    YOLODFLFreeSegPost,
)
from .yolo_nmsfree_post import YOLONMSFreeDetectionPost


def build_postprocess(
    pre_cfg: dict,
    post_cfg: dict,
    **kwargs: object,
) -> PostBase:
    """Builds a postprocessing object based on the model configuration.

    Args:
        pre_cfg (dict): Preprocessing configuration from the model info.
        post_cfg (dict): Postprocessing configuration from the model info.
            Must contain "task" and relevant flags for the specific task.
        **kwargs: Optional runtime overrides passed to the postprocessor.

    Returns:
        PostBase: An instance of a postprocessing class tailored for the task.

    Raises:
        NotImplementedError: If the specified task is not supported.
    """
    task_lower = post_cfg["task"].lower()
    if task_lower == "image_classification":
        return ClsPost(pre_cfg, post_cfg)
    if task_lower == "depth_estimation":
        return DepthPost(pre_cfg, post_cfg)
    if task_lower == "semantic_segmentation":
        return SemanticSegPost(pre_cfg, post_cfg)
    if task_lower in {"object_detection", "face_detection"}:
        if post_cfg.get("anchors", False):
            return YOLOAnchorDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("dflfree", False):  # nms free is only available for detection
            return YOLODFLFreeDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("nmsfree", False):
            return YOLONMSFreeDetectionPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessDetectionPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task_lower == "instance_segmentation":
        if post_cfg.get("anchors", False):
            return YOLOAnchorSegPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        if post_cfg.get("dflfree", False):
            return YOLODFLFreeSegPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessSegPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task_lower == "pose_estimation":
        if post_cfg.get("dflfree", False):
            return YOLODFLFreePosePost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessPosePost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    if task_lower == "obb":
        if post_cfg.get("dflfree", False):
            return YOLODFLFreeOBBPost(
                pre_cfg,
                post_cfg,
                **kwargs,
            )
        return YOLOAnchorlessOBBPost(
            pre_cfg,
            post_cfg,
            **kwargs,
        )
    raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")
