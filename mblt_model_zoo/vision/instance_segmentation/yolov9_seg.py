"""
YOLOv9 Segmentation model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv9cSeg_Set(ModelInfoSet):
    """Configuration set for YOLOv9cSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv9c-seg",
            "filename": "yolov9c-seg.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLOv9eSeg_Set(ModelInfoSet):
    """Configuration set for YOLOv9eSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv9e-seg",
            "filename": "yolov9e-seg.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


def YOLOv9cSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv9cSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv9cSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv9eSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv9eSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv9eSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
