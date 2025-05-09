from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv9m_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_detection/yolov9m.mxq",
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


class YOLOv9c_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_detection/yolov9c.mxq",
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


class YOLOv9m(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9m_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9m_Set. Available types: {YOLOv9m_Set.__dict__.keys()}"
        model_cfg = YOLOv9m_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        pre_cfg = YOLOv9m_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9m_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9c(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9c_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9c_Set. Available types: {YOLOv9c_Set.__dict__.keys()}"
        model_cfg = YOLOv9c_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        pre_cfg = YOLOv9c_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9c_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
