from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class Regnet_X_16GF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_16gf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_1_6GF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_1_6gf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_32GF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_32gf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_3_2GF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_3_2gf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_400MF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_400mf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_800MF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_800mf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_8GF_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/aries/global/vision/image_classification/regnet_x_8gf_torchvision.mxq",
                },
                "regulus": {"single": None},
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


class Regnet_X_16GF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_16GF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_16GF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_16GF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_16GF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_16GF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_1_6GF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_1_6GF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_1_6GF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_1_6GF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_1_6GF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_1_6GF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_32GF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_32GF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_32GF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_32GF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_32GF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_32GF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_3_2GF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_3_2GF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_3_2GF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_3_2GF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_3_2GF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_3_2GF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_400MF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_400MF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_400MF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_400MF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_400MF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_400MF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_800MF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_800MF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_800MF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_800MF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_800MF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_800MF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class Regnet_X_8GF(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in Regnet_X_8GF_Set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {Regnet_X_8GF_Set.__dict__.keys()}"
        model_cfg = Regnet_X_8GF_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = Regnet_X_8GF_Set.__dict__[model_type].value.pre_cfg
        post_cfg = Regnet_X_8GF_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
