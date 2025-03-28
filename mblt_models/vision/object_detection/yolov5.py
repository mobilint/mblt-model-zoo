from mblt_models.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_models.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class YOLOv5n_Set(ModelInfoSet):
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
        },
    )
