from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO11sSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11s-seg/aries/single/yolo11s-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11s-seg/aries/multi/yolo11s-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11s-seg/aries/global/yolo11s-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLO11mSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11m-seg/aries/single/yolo11m-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11m-seg/aries/multi/yolo11m-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11m-seg/aries/global/yolo11m-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLO11lSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11l-seg/aries/single/yolo11l-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11l-seg/aries/multi/yolo11l-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11l-seg/aries/global/yolo11l-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLO11xSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11x-seg/aries/single/yolo11x-seg.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11x-seg/aries/multi/yolo11x-seg.mxq",
                    "global": "https://dl.mobilint.com/model/vision/instance_segmentation/yolo11x-seg/aries/global/yolo11x-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


def YOLO11sSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11sSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11mSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11mSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11lSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11lSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11xSeg(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11xSeg_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
