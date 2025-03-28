from mblt_models.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_models.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class ResNet18_Set(ModelInfoSet):
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
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGNET1K_V1  # Default model


class ResNet34_Set(ModelInfoSet):
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
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGNET1K_V1  # Default model


class ResNet50_Set(ModelInfoSet):
    IMAGNET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://maccel.mobilint.com/resnet50.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url": "https://maccel.mobilint.com/resnet50.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGNET1K_V1  # Default model


class ResNet50(MBLT_Engine):
    def __init__(
        self,
    ):
        model_cfg = ResNet50_Set.DEFAULT.value.model_cfg
        pre_cfg = ResNet50_Set.DEFAULT.value.pre_cfg
        post_cfg = ResNet50_Set.DEFAULT.value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
