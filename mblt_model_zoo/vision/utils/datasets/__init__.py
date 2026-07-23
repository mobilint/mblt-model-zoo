"""
Datasets utilities and loaders.
"""

from __future__ import annotations

from .coco import (
    get_coco_class_num,
    get_coco_det_palette,
    get_coco_inv,
    get_coco_keypoint_palette,
    get_coco_label,
    get_coco_limb_palette,
    get_coco_pose_skeleton,
)
from .dataloader import (
    CustomADE20K,
    CustomCocodata,
    CustomDOTAv1,
    CustomImageFolder,
    CustomNYUDepth,
    CustomWiderface,
    get_ade20k_loader,
    get_coco_loader,
    get_dota_loader,
    get_imagenet_loader,
    get_nyu_depth_loader,
    get_widerface_loader,
)
from .dotav1 import get_dotav1_class_num, get_dotav1_label
from .imagenet import get_imagenet_label
from .organizer import (
    organize_ade20k,
    organize_coco,
    organize_dotav1,
    organize_imagenet,
    organize_nyu_depth,
    organize_widerface,
)

__all__: list[str] = [
    "get_coco_class_num",
    "get_coco_det_palette",
    "get_coco_inv",
    "get_coco_keypoint_palette",
    "get_coco_label",
    "get_coco_limb_palette",
    "get_coco_pose_skeleton",
    "get_dotav1_class_num",
    "get_dotav1_label",
    "CustomADE20K",
    "CustomCocodata",
    "CustomDOTAv1",
    "CustomImageFolder",
    "CustomNYUDepth",
    "CustomWiderface",
    "get_ade20k_loader",
    "get_coco_loader",
    "get_dota_loader",
    "get_imagenet_loader",
    "get_nyu_depth_loader",
    "get_widerface_loader",
    "get_imagenet_label",
    "organize_coco",
    "organize_ade20k",
    "organize_dotav1",
    "organize_imagenet",
    "organize_nyu_depth",
    "organize_widerface",
]
