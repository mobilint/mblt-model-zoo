from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv10b_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov10b/aries/single/yolov10b.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov10b/aries/multi/yolov10b.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov10b/aries/global/yolov10b.mxq",
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
            "nmsfree": True,  # nms free yolo
        },
    )
    DEFAULT = COCO_V1


def YOLOv10b(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv10b_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
