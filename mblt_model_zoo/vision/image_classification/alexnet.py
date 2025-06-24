from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class AlexNet_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "single": None,
                "multi": None,
                "global": "https://dl.mobilint.com/model/image_classification/alexnet_torchvision.mxq",
                "regulus": None,
            },
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
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
    ):
        assert (
            model_type in AlexNet_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {AlexNet_Set.__dict__.keys()}"
        model_cfg = AlexNet_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        pre_cfg = AlexNet_Set.__dict__[model_type].value.pre_cfg
        post_cfg = AlexNet_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
