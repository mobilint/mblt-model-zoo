"""
Postprocessing builder.
"""

from .base import PostBase
from .cls_post import ClsPost
from .yolo_anchor_post import YOLOAnchorPost, YOLOAnchorSegPost
from .yolo_anchorless_post import (
    YOLOAnchorlessPosePost,
    YOLOAnchorlessPost,
    YOLOAnchorlessSegPost,
)
from .yolo_nmsfree_post import YOLONMSFreePosePost, YOLONMSFreePost, YOLONMSFreeSegPost


def build_postprocess(
    pre_cfg: dict,
    post_cfg: dict,
) -> PostBase:
    """Build the postprocess object.

    Args:
        pre_cfg (dict): Preprocessing configuration.
        post_cfg (dict): Postprocessing configuration.

    Returns:
        PostBase: The postprocess object.
    """
    task_lower = post_cfg["task"].lower()
    if task_lower == "image_classification":
        return ClsPost(pre_cfg, post_cfg)
    if task_lower == "object_detection":
        if post_cfg.get("anchors", False):
            return YOLOAnchorPost(
                pre_cfg,
                post_cfg,
            )
        if post_cfg.get("nmsfree", False):  # nms free is only available for detection
            return YOLONMSFreePost(
                pre_cfg,
                post_cfg,
            )
        return YOLOAnchorlessPost(
            pre_cfg,
            post_cfg,
        )

    if task_lower == "instance_segmentation":
        if post_cfg.get("anchors", False):
            return YOLOAnchorSegPost(
                pre_cfg,
                post_cfg,
            )
        if post_cfg.get("nmsfree", False):
            return YOLONMSFreeSegPost(
                pre_cfg,
                post_cfg,
            )
        return YOLOAnchorlessSegPost(
            pre_cfg,
            post_cfg,
        )

    if task_lower == "pose_estimation":
        if post_cfg.get("nmsfree", False):
            return YOLONMSFreePosePost(
                pre_cfg,
                post_cfg,
            )
        return YOLOAnchorlessPosePost(
            pre_cfg,
            post_cfg,
        )

    raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")
