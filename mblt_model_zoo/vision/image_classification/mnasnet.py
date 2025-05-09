from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class MNasNet0_5_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/mnasnet0_5_torchvision.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class MNasNet1_0_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/mnasnet1_0_torchvision.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class MNasNet0_75_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/mnasnet0_75_torchvision.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class MNasNet0_5(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in MNasNet0_5_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {MNasNet0_5_Set.__dict__.keys()}"
        model_cfg = MNasNet0_5_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = MNasNet0_5_Set.__dict__[model_type].value.pre_cfg
        post_cfg = MNasNet0_5_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class MNasNet0_75(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in MNasNet0_75_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {MNasNet0_75_Set.__dict__.keys()}"
        model_cfg = MNasNet0_75_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = MNasNet0_75_Set.__dict__[model_type].value.pre_cfg
        post_cfg = MNasNet0_75_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class MNasNet1_0(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in MNasNet1_0_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {MNasNet1_0_Set.__dict__.keys()}"
        model_cfg = MNasNet1_0_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = MNasNet1_0_Set.__dict__[model_type].value.pre_cfg
        post_cfg = MNasNet1_0_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
