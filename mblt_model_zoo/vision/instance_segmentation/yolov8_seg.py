from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8sSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8s-seg/aries/single/yolov8s-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8s-seg/aries/multi/yolov8s-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8s-seg/aries/global/yolov8s-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8mSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8m-seg/aries/single/yolov8m-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8m-seg/aries/multi/yolov8m-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8m-seg/aries/global/yolov8m-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8lSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8l-seg/aries/single/yolov8l-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8l-seg/aries/multi/yolov8l-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8l-seg/aries/global/yolov8l-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8xSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8x-seg/aries/single/yolov8x-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8x-seg/aries/multi/yolov8x-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolov8x-seg/aries/global/yolov8x-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


def YOLOv8sSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8sSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8mSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8mSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8lSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8lSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8xSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8xSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
