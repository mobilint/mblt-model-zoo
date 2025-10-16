from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNext50_32x4d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V1/aries/single/resnext50_32x4d_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V1/aries/multi/resnext50_32x4d_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V1/aries/global/resnext50_32x4d_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V2/aries/single/resnext50_32x4d_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V2/aries/multi/resnext50_32x4d_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnext50_32x4d_IMAGENET1K_V2/aries/global/resnext50_32x4d_IMAGENET1K_V2.mxq",
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
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNext101_32x8d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V1/aries/single/resnext101_32x8d_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V1/aries/multi/resnext101_32x8d_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V1/aries/global/resnext101_32x8d_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V2/aries/single/resnext101_32x8d_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V2/aries/multi/resnext101_32x8d_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnext101_32x8d_IMAGENET1K_V2/aries/global/resnext101_32x8d_IMAGENET1K_V2.mxq",
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
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNext101_64x4d_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnext101_64x4d_IMAGENET1K_V1/aries/single/resnext101_64x4d_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnext101_64x4d_IMAGENET1K_V1/aries/multi/resnext101_64x4d_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnext101_64x4d_IMAGENET1K_V1/aries/global/resnext101_64x4d_IMAGENET1K_V1.mxq",
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


def ResNext50_32x4d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ResNext50_32x4d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNext101_32x8d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ResNext101_32x8d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNext101_64x4d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ResNext101_64x4d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
