from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ShuffleNet_V2_X1_0_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_0_IMAGENET1K_V1/aries/single/shufflenet_v2_x1_0_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_0_IMAGENET1K_V1/aries/multi/shufflenet_v2_x1_0_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_0_IMAGENET1K_V1/aries/global/shufflenet_v2_x1_0_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
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
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ShuffleNet_V2_X1_5_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_5_IMAGENET1K_V1/aries/single/shufflenet_v2_x1_5_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_5_IMAGENET1K_V1/aries/multi/shufflenet_v2_x1_5_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x1_5_IMAGENET1K_V1/aries/global/shufflenet_v2_x1_5_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x2_0_IMAGENET1K_V1/aries/single/shufflenet_v2_x2_0_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x2_0_IMAGENET1K_V1/aries/multi/shufflenet_v2_x2_0_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/shufflenet_v2_x2_0_IMAGENET1K_V1/aries/global/shufflenet_v2_x2_0_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
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


def ShuffleNet_V2_X1_0(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ShuffleNet_V2_X1_0_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ShuffleNet_V2_X1_5(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ShuffleNet_V2_X1_5_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ShuffleNet_V2_X2_0(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ShuffleNet_V2_X2_0_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
