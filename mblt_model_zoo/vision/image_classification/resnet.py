from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNet18_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet34_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet50_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
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
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet101_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url": "/",
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


class ResNet152_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url": "/",
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


class ResNet18(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNet18_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNet18_Set.__dict__.keys()}"
        model_cfg = ResNet18_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNet18_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNet18_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet34(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNet34_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNet34_Set.__dict__.keys()}"
        model_cfg = ResNet34_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNet34_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNet34_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet50(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNet50_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNet50_Set.__dict__.keys()}"
        model_cfg = ResNet50_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNet50_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNet50_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet101(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNet101_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNet101_Set.__dict__.keys()}"
        model_cfg = ResNet101_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNet101_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNet101_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet152(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ResNet152_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ResNet152_Set.__dict__.keys()}"
        model_cfg = ResNet152_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model

        pre_cfg = ResNet152_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ResNet152_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
