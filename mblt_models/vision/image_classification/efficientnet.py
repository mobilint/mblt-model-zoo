from mblt_models.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_models.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class EfficientNet_B0_Set(ModelInfoSet):
    IMAGNET1K_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGNET1K_V1  # Default model


class EfficientNet_B1_Set(ModelInfoSet):
    IMAGNET1K_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [240, 240],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )
    IMAGNET1K_V2 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 255,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [240, 240],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )

    DEFAULT = IMAGNET1K_V1  # Default model
