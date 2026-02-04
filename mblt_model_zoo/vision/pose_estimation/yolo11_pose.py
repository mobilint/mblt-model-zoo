"""
YOLO11 Pose Estimation model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO11nPose_Set(ModelInfoSet):
    """Configuration set for YOLO11nPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO11nPose",
            "filename": "yolo11n-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLO11sPose_Set(ModelInfoSet):
    """Configuration set for YOLO11sPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO11sPose",
            "filename": "yolo11s-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLO11mPose_Set(ModelInfoSet):
    """Configuration set for YOLO11mPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO11mPose",
            "filename": "yolo11m-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLO11lPose_Set(ModelInfoSet):
    """Configuration set for YOLO11lPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO11lPose",
            "filename": "yolo11l-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLO11xPose_Set(ModelInfoSet):
    """Configuration set for YOLO11xPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO11xPose",
            "filename": "yolo11x-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


def YOLO11nPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO11nPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO11nPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11sPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO11sPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO11sPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11mPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO11mPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO11mPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11lPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO11lPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO11lPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11xPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLO11xPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLO11xPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
