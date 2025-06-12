from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv7_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_detection/yolov7_640_640.mxq",
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


class YOLOv7(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv7_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8m_Set. Available types: {YOLOv7_Set.__dict__.keys()}"
        model_cfg = YOLOv7_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        pre_cfg = YOLOv7_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv7_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
