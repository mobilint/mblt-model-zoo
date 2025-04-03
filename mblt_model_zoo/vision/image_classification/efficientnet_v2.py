from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class EfficientNet_V2_S_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 384,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class EfficientNet_V2_S(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in EfficientNet_V2_S_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {EfficientNet_V2_S_Set.__dict__.keys()}"
        model_cfg = EfficientNet_V2_S_Set.__dict__[model_type].model_cfg
        if local_model:
            model_cfg["url"] = local_model
        pre_cfg = EfficientNet_V2_S_Set.__dict__[model_type].pre_cfg
        post_cfg = EfficientNet_V2_S_Set.__dict__[model_type].post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
