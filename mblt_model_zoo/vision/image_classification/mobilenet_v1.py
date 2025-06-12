from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class MobileNet_V1_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url": "https://dl.mobilint.com/model/image_classification/mobilenet_v1.mxq",
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


class MobileNet_V1(MBLT_Engine):
    def __init__(self, local_path: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in MobileNet_V1_Set.__dict__.keys()
        ), f"Model type {model_type} not found. Available types: {MobileNet_V1_Set.__dict__.keys()}"
        model_cfg = MobileNet_V1_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path

        pre_cfg = MobileNet_V1_Set.__dict__[model_type].value.pre_cfg
        post_cfg = MobileNet_V1_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
