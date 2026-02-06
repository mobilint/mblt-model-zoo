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
from .yolo_dflfree_post import YOLODFLFreePosePost, YOLODFLFreePost, YOLODFLFreeSegPost
from .yolo_nmsfree_post import YOLONMSFreePost


def build_postprocess(
    pre_cfg: dict,
    post_cfg: dict,
) -> PostBase:
    """Builds a postprocessing object based on the model configuration.

    Args:
        pre_cfg (dict): Preprocessing configuration from the model info.
        post_cfg (dict): Postprocessing configuration from the model info.
            Must contain "task" and relevant flags for the specific task.

    Returns:
        PostBase: An instance of a postprocessing class tailored for the task.

    Raises:
        NotImplementedError: If the specified task is not supported.
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
        if post_cfg.get("dflfree", False):  # nms free is only available for detection
            return YOLODFLFreePost(
                pre_cfg,
                post_cfg,
            )
        if post_cfg.get("nmsfree", False):
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
        if post_cfg.get("dflfree", False):
            return YOLODFLFreeSegPost(
                pre_cfg,
                post_cfg,
            )
        return YOLOAnchorlessSegPost(
            pre_cfg,
            post_cfg,
        )
    if task_lower == "pose_estimation":
        if post_cfg.get("dflfree", False):
            return YOLODFLFreePosePost(
                pre_cfg,
                post_cfg,
            )
        return YOLOAnchorlessPosePost(
            pre_cfg,
            post_cfg,
        )
    raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")
