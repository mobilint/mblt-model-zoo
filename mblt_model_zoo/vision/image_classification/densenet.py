from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class DenseNet121_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "single": None,
                "multi": None,
                "global": "https://dl.mobilint.com/model/image_classification/densenet121_torchvision.mxq",
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


class DenseNet161_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "single": None,
                "multi": None,
                "global": "https://dl.mobilint.com/model/image_classification/densenet161_torchvision.mxq",
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


class DenseNet169_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "single": None,
                "multi": None,
                "global": "https://dl.mobilint.com/model/image_classification/densenet169_torchvision.mxq",
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


class DenseNet201_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "single": None,
                "multi": None,
                "global": "https://dl.mobilint.com/model/image_classification/densenet201_torchvision.mxq",
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


class DenseNet121(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
    ):
        assert (
            model_type in DenseNet121_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {DenseNet121_Set.__dict__.keys()}"
        model_cfg = DenseNet121_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        pre_cfg = DenseNet121_Set.__dict__[model_type].value.pre_cfg
        post_cfg = DenseNet121_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DenseNet161(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
    ):
        assert (
            model_type in DenseNet161_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {DenseNet161_Set.__dict__.keys()}"
        model_cfg = DenseNet161_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        pre_cfg = DenseNet161_Set.__dict__[model_type].value.pre_cfg
        post_cfg = DenseNet161_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DenseNet169(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
    ):
        assert (
            model_type in DenseNet169_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {DenseNet169_Set.__dict__.keys()}"
        model_cfg = DenseNet169_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        pre_cfg = DenseNet169_Set.__dict__[model_type].value.pre_cfg
        post_cfg = DenseNet169_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DenseNet201(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
    ):
        assert (
            model_type in DenseNet201_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {DenseNet201_Set.__dict__.keys()}"
        model_cfg = DenseNet201_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        pre_cfg = DenseNet201_Set.__dict__[model_type].value.pre_cfg
        post_cfg = DenseNet201_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
