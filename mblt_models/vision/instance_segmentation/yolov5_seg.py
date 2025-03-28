from mblt_models.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_models.vision.wrapper import MBLT_Engine
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


class YOLOv5mSeg(MBLT_Engine):
    def __init__(self, local_model: str = None):
        model_cfg = YOLOv5mSeg_Set.DEFAULT.value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv5mSeg_Set.DEFAULT.value.pre_cfg
        post_cfg = YOLOv5mSeg_Set.DEFAULT.value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
