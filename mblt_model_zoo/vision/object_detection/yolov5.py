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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5s6u_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5s6u/aries/single/yolov5s6u.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5s6u/aries/multi/yolov5s6u.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5s6u/aries/global/yolov5s6u.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 4,  # Number of detection layers
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5m6u_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5m6u/aries/single/yolov5m6u.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5m6u/aries/multi/yolov5m6u.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5m6u/aries/global/yolov5m6u.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 4,  # Number of detection layers
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5l6_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6/aries/single/yolov5l6.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6/aries/multi/yolov5l6.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6/aries/global/yolov5l6.mxq",
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
            "SetOrder": {"shape": "CHW"},
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


class YOLOv5l6u_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6u/aries/single/yolov5l6u.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6u/aries/multi/yolov5l6u.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5l6u/aries/global/yolov5l6u.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 4,  # Number of detection layers
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv5x6_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6/aries/single/yolov5x6.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6/aries/multi/yolov5x6.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6/aries/global/yolov5x6.mxq",
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
            "SetOrder": {"shape": "CHW"},
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


class YOLOv5x6u_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6u/aries/single/yolov5x6u.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6u/aries/multi/yolov5x6u.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov5x6u/aries/global/yolov5x6u.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 4,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


def YOLOv5su(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5su_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5s6u(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5s6u_Set,
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
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5mu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5m6u(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5m6u_Set,
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
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5lu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5l6(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5l6_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5l6u(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5l6u_Set,
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
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5xu_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5x6(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5x6_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv5x6u(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv5x6u_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
