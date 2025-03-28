from .base import PostBase
from .cls_post import ClsPost
from .yolo_anchor_post import *
from .yolo_anchorless_post import *


def build_postprocess(pre_cfg: dict, post_cfg: dict) -> PostBase:
    task_lower = post_cfg["task"].lower()
    if task_lower == "image_classification":
        return ClsPost(pre_cfg, post_cfg)
    elif task_lower == "object_detection":
        if post_cfg.get("anchor") is not None:
            return YOLOAnchorPost(pre_cfg, post_cfg)
        else:
            return YOLOAnchorlessPost(pre_cfg, post_cfg)

    elif task_lower == "instance_segmentation":
        if post_cfg.get("anchor") is not None:
            return YOLOAnchorSegPost(pre_cfg, post_cfg)
        else:
            return YOLOAnchorlessSegPost(pre_cfg, post_cfg)

    elif task_lower == "pose_estimation":
        if post_cfg.get("anchor") is not None:
            raise NotImplementedError(
                "Pose estimation with anchor is not implemented yet"
            )
        elif post_cfg.get("anchorless") is not None:
            return YOLOAnchorlessPosePost(pre_cfg, post_cfg)

    else:
        raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")
