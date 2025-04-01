from mblt_model_zoo.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_model_zoo.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class YOLOv9cSeg_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
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


class YOLOv9eSeg_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
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


class YOLOv9cSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9cSeg_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv9cSeg_Set. Available types: {YOLOv9cSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv9cSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9cSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9cSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9eSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9eSeg_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv9eSeg_Set. Available types: {YOLOv9eSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv9eSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9eSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9eSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
