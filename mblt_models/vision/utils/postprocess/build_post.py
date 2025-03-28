from .base import PostBase
from .cls_post import ClsPost


def build_postprocess(pre_cfg: dict, post_cfg: dict) -> PostBase:
    if post_cfg["task"] == "image_classification":
        return ClsPost(pre_cfg, post_cfg)
    else:
        raise NotImplementedError(f"Task {post_cfg['task']} is not implemented yet")
