from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv7_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/single/yolov7.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/multi/yolov7.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7/aries/global/yolov7.mxq",
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
            "anchors": [
                [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv7x_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/single/yolov7x.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/multi/yolov7x.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov7x/aries/global/yolov7x.mxq",
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
    return MBLT_Engine.from_model_info_set(
        YOLOv7_Set,
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
    return MBLT_Engine.from_model_info_set(
        YOLOv7x_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
