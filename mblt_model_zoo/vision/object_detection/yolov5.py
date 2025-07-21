from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv5su_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5su/aries/single/yolov5su.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5su/aries/multi/yolov5su.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5su/aries/global/yolov5su.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5mu_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5mu/aries/single/yolov5mu.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5mu/aries/multi/yolov5mu.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5mu/aries/global/yolov5mu.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5lu_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5lu/aries/single/yolov5lu.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5lu/aries/multi/yolov5lu.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5lu/aries/global/yolov5lu.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5xu_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5xu/aries/single/yolov5xu.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5xu/aries/multi/yolov5xu.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5xu/aries/global/yolov5xu.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


def YOLOv5su(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
):
    return MBLT_Engine.from_model_info_set(
        YOLOv5su_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5mu(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
):
    return MBLT_Engine.from_model_info_set(
        YOLOv5mu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5lu(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
):
    return MBLT_Engine.from_model_info_set(
        YOLOv5lu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5xu(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
):
    return MBLT_Engine.from_model_info_set(
        YOLOv5xu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
