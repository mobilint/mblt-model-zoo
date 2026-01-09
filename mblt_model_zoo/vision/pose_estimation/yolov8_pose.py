from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8nPose_Set(ModelInfoSet):
    """Model information set for YOLOv8n-pose."""

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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8sPose_Set(ModelInfoSet):
    """Model information set for YOLOv8s-pose."""

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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8mPose_Set(ModelInfoSet):
    """Model information set for YOLOv8m-pose."""

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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8lPose_Set(ModelInfoSet):
    """Model information set for YOLOv8l-pose."""

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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8xPose_Set(ModelInfoSet):
    """Model information set for YOLOv8x-pose."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/pose_estimation/yolov8x-pose/aries/single/yolov8x-pose.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/pose_estimation/yolov8x-pose/aries/multi/yolov8x-pose.mxq",
                    "global": "https://dl.mobilint.com/model/vision/pose_estimation/yolov8x-pose/aries/global/yolov8x-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


def YOLOv8nPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Create an MBLT_Engine for YOLOv8n-pose.

    Args:
        local_path (str, optional): Path to local model file.
        model_type (str): Type of model (e.g., 'DEFAULT').
        infer_mode (str): Inference mode (e.g., 'global').
        product (str): Target product (e.g., 'aries').

    Returns:
        MBLT_Engine: The pose estimation engine.
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
    """
    Create an MBLT_Engine for YOLOv8s-pose.

    Args:
        local_path (str, optional): Path to local model file.
        model_type (str): Type of model (e.g., 'DEFAULT').
        infer_mode (str): Inference mode (e.g., 'global').
        product (str): Target product (e.g., 'aries').

    Returns:
        MBLT_Engine: The pose estimation engine.
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
    """
    Create an MBLT_Engine for YOLOv8m-pose.

    Args:
        local_path (str, optional): Path to local model file.
        model_type (str): Type of model (e.g., 'DEFAULT').
        infer_mode (str): Inference mode (e.g., 'global').
        product (str): Target product (e.g., 'aries').

    Returns:
        MBLT_Engine: The pose estimation engine.
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
    """
    Create an MBLT_Engine for YOLOv8l-pose.

    Args:
        local_path (str, optional): Path to local model file.
        model_type (str): Type of model (e.g., 'DEFAULT').
        infer_mode (str): Inference mode (e.g., 'global').
        product (str): Target product (e.g., 'aries').

    Returns:
        MBLT_Engine: The pose estimation engine.
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
    """
    Create an MBLT_Engine for YOLOv8x-pose.

    Args:
        local_path (str, optional): Path to local model file.
        model_type (str): Type of model (e.g., 'DEFAULT').
        infer_mode (str): Inference mode (e.g., 'global').
        product (str): Target product (e.g., 'aries').

    Returns:
        MBLT_Engine: The pose estimation engine.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv8xPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
