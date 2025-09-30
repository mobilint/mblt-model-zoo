from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8xPose_Set(ModelInfoSet):
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


def YOLOv8xPose(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8xPose_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
