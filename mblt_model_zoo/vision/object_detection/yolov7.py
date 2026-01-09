from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv7_Set(ModelInfoSet):
    """YOLOv7 model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/single/yolov7.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/multi/yolov7.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/global/yolov7.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7d6_Set(ModelInfoSet):
    """YOLOv7d6 model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7d6/aries/single/yolov7d6.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7d6/aries/multi/yolov7d6.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7d6/aries/global/yolov7d6.mxq",
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
                "img_size": [1280, 1280],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [19, 27, 44, 40, 38, 94],  # P3/8
                [96, 68, 86, 152, 180, 137],  # P4/16
                [140, 301, 303, 264, 238, 542],  # P5/32
                [436, 615, 739, 380, 925, 792],  # P6/64
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7e6_Set(ModelInfoSet):
    """YOLOv7e6 model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6/aries/single/yolov7e6.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6/aries/multi/yolov7e6.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6/aries/global/yolov7e6.mxq",
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
                "img_size": [1280, 1280],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [19, 27, 44, 40, 38, 94],  # P3/8
                [96, 68, 86, 152, 180, 137],  # P4/16
                [140, 301, 303, 264, 238, 542],  # P5/32
                [436, 615, 739, 380, 925, 792],  # P6/64
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7e6e_Set(ModelInfoSet):
    """YOLOv7e6e model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6e/aries/single/yolov7e6e.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6e/aries/multi/yolov7e6e.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7e6e/aries/global/yolov7e6e.mxq",
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
                "img_size": [1280, 1280],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [19, 27, 44, 40, 38, 94],  # P3/8
                [96, 68, 86, 152, 180, 137],  # P4/16
                [140, 301, 303, 264, 238, 542],  # P5/32
                [436, 615, 739, 380, 925, 792],  # P6/64
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7w6_Set(ModelInfoSet):
    """YOLOv7w6 model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7w6/aries/single/yolov7w6.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7w6/aries/multi/yolov7w6.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7w6/aries/global/yolov7w6.mxq",
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
                "img_size": [1280, 1280],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [19, 27, 44, 40, 38, 94],  # P3/8
                [96, 68, 86, 152, 180, 137],  # P4/16
                [140, 301, 303, 264, 238, 542],  # P5/32
                [436, 615, 739, 380, 925, 792],  # P6/64
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7x_Set(ModelInfoSet):
    """YOLOv7x model information set."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/single/yolov7x.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/multi/yolov7x.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/global/yolov7x.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
            ],
        },
    )
    DEFAULT = COCO_V1


def YOLOv7(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv7d6(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7d6 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7d6_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv7e6(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7e6 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7e6_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv7e6e(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7e6e model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7e6e_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv7w6(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7w6 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7w6_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv7x(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the YOLOv7x model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "COCO_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        YOLOv7x_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
