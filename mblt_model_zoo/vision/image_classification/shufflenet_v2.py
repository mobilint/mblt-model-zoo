from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ShuffleNet_V2_X1_0_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/shufflenet_v2_x1_0_torchvision.mxq",
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


class ShuffleNet_V2_X1_5_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/shufflenet_v2_x1_5_torchvision.mxq",
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


class ShuffleNet_V2_X2_0_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/shufflenet_v2_x2_0_torchvision.mxq",
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


class ShuffleNet_V2_X1_0(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ShuffleNet_V2_X1_0_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ShuffleNet_V2_X1_0_Set.__dict__.keys()}"
        model_cfg = ShuffleNet_V2_X1_0_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = ShuffleNet_V2_X1_0_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ShuffleNet_V2_X1_0_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ShuffleNet_V2_X1_5(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ShuffleNet_V2_X1_5_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ShuffleNet_V2_X1_5_Set.__dict__.keys()}"
        model_cfg = ShuffleNet_V2_X1_5_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = ShuffleNet_V2_X1_5_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ShuffleNet_V2_X1_5_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ShuffleNet_V2_X2_0(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in ShuffleNet_V2_X2_0_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {ShuffleNet_V2_X2_0_Set.__dict__.keys()}"
        model_cfg = ShuffleNet_V2_X2_0_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = ShuffleNet_V2_X2_0_Set.__dict__[model_type].value.pre_cfg
        post_cfg = ShuffleNet_V2_X2_0_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
