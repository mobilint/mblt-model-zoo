from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNext50_32x4d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_classification/resnext50_32x4d_torchvision.mxq",
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNext101_32x8d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_classification/resnext101_32x8d_torchvision.mxq",
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNext101_64x4d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_classification/resnext101_64x4d_torchvision.mxq",
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNext50_32x4d(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNext50_32x4d_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNext50_32x4d_Set.__dict__.keys()}"
        model_cfg = ResNext50_32x4d_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNext50_32x4d_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNext50_32x4d_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNext101_32x8d(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNext101_32x8d_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNext101_32x8d_Set.__dict__.keys()}"
        model_cfg = ResNext101_32x8d_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNext101_32x8d_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNext101_32x8d_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNext101_64x4d(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNext101_64x4d_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNext101_64x4d_Set.__dict__.keys()}"
        model_cfg = ResNext101_64x4d_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNext101_64x4d_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNext101_64x4d_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
