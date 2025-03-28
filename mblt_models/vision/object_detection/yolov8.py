from mblt_models.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_models.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class YOLOv8n_Set(ModelInfoSet):
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
        },
    )
