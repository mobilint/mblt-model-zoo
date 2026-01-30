"""
YOLO26 Segmentation model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO26nSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26nSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26sSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26sSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26mSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26mSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26lSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26lSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26xSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26xSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


def YOLO26nSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a YOLO26nSeg model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26nSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO26sSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a YOLO26sSeg model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26sSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO26mSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a YOLO26mSeg model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26mSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO26lSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a YOLO26lSeg model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26lSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO26xSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a YOLO26xSeg model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26xSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
