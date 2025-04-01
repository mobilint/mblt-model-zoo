from mblt_model_zoo.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_model_zoo.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class YOLOv5nSeg_Set(ModelInfoSet):
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
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )


class YOLOv5sSeg_Set(ModelInfoSet):
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
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )


class YOLOv5mSeg_Set(ModelInfoSet):
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
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )


class YOLOv5lSeg_Set(ModelInfoSet):
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
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )


class YOLOv5xSeg_Set(ModelInfoSet):
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
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )


class YOLOv5nSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv5nSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5nSeg_Set. Available types: {YOLOv5nSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5nSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5nSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5nSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv5sSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv5sSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5sSeg_Set. Available types: {YOLOv5sSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5sSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5sSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5sSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv5mSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv5mSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5mSeg_Set. Available types: {YOLOv5mSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5mSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5mSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5mSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv5lSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv5lSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5lSeg_Set. Available types: {YOLOv5lSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv5xSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv5xSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5xSeg_Set. Available types: {YOLOv5xSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5xSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5xSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5xSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
