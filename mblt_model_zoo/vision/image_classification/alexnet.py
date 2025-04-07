from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class AlexNet_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/alexnet_torchvision.mxq",
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


class AlexNet(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in AlexNet_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {AlexNet_Set.__dict__.keys()}"
        model_cfg = AlexNet_Set.__dict__[model_type].value.model_cfg
        if local_model:
            model_cfg["url"] = local_model
        pre_cfg = AlexNet_Set.__dict__[model_type].value.pre_cfg
        post_cfg = AlexNet_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
