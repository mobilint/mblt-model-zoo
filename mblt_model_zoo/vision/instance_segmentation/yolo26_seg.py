"""
YOLO26 Segmentation model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO26nSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26nSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26n-seg",
            "filename": "yolo26n-seg.mxq",
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
            "dflfree": True,  # dfl free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26sSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26sSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={"repo_id": "mobilint/YOLO26s-seg", "filename": "yolo26s-seg.mxq"},
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
            "dflfree": True,  # dfl free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26mSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26mSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={"repo_id": "mobilint/YOLO26m-seg", "filename": "yolo26m-seg.mxq"},
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
            "dflfree": True,  # dfl free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26lSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26lSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={"repo_id": "mobilint/YOLO26l-seg", "filename": "yolo26l-seg.mxq"},
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
            "dflfree": True,  # dfl free yolo
        },
    )
    DEFAULT = COCO_V1


class YOLO26xSeg_Set(ModelInfoSet):
    """Configuration set for YOLO26xSeg models."""

    COCO_V1 = ModelInfo(
        model_cfg={"repo_id": "mobilint/YOLO26x-seg", "filename": "yolo26x-seg.mxq"},
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
            "dflfree": True,  # dfl free yolo
        },
    )
    DEFAULT = COCO_V1


def YOLO26nSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO26nSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
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
    """Constructs a YOLO26sSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
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
    """Constructs a YOLO26mSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
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
    """Constructs a YOLO26lSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
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
    """Constructs a YOLO26xSeg model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO26xSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
