"""
YOLOv8 Pose Estimation model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8nPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8nPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8nPose",
            "filename": "yolov8n-pose.mxq",
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


class YOLOv8sPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8sPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8sPose",
            "filename": "yolov8s-pose.mxq",
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


class YOLOv8mPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8mPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8mPose",
            "filename": "yolov8m-pose.mxq",
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


class YOLOv8lPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8lPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8lPose",
            "filename": "yolov8l-pose.mxq",
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


class YOLOv8xPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8xPose models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8xPose",
            "filename": "yolov8x-pose.mxq",
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


def YOLOv8nPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv8nPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8nPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8sPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv8sPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8sPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8mPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv8mPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8mPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8lPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv8lPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8lPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8xPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a YOLOv8xPose model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8xPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
