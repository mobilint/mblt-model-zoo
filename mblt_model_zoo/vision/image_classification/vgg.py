from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class VGG11_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg11_IMAGENET1K_V1/aries/single/vgg11_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg11_IMAGENET1K_V1/aries/multi/vgg11_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg11_IMAGENET1K_V1/aries/global/vgg11_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG11_BN_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg11_bn_IMAGENET1K_V1/aries/single/vgg11_bn_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg11_bn_IMAGENET1K_V1/aries/multi/vgg11_bn_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg11_bn_IMAGENET1K_V1/aries/global/vgg11_bn_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG13_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg13_IMAGENET1K_V1/aries/single/vgg13_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg13_IMAGENET1K_V1/aries/multi/vgg13_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg13_IMAGENET1K_V1/aries/global/vgg13_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG13_BN_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg13_bn_IMAGENET1K_V1/aries/single/vgg13_bn_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg13_bn_IMAGENET1K_V1/aries/multi/vgg13_bn_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg13_bn_IMAGENET1K_V1/aries/global/vgg13_bn_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG16_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg16_IMAGENET1K_V1/aries/single/vgg16_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg16_IMAGENET1K_V1/aries/multi/vgg16_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg16_IMAGENET1K_V1/aries/global/vgg16_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG16_BN_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg16_bn_IMAGENET1K_V1/aries/single/vgg16_bn_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg16_bn_IMAGENET1K_V1/aries/multi/vgg16_bn_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg16_bn_IMAGENET1K_V1/aries/global/vgg16_bn_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG19_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg19_IMAGENET1K_V1/aries/single/vgg19_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg19_IMAGENET1K_V1/aries/multi/vgg19_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg19_IMAGENET1K_V1/aries/global/vgg19_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class VGG19_BN_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vgg19_bn_IMAGENET1K_V1/aries/single/vgg19_bn_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vgg19_bn_IMAGENET1K_V1/aries/multi/vgg19_bn_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vgg19_bn_IMAGENET1K_V1/aries/global/vgg19_bn_IMAGENET1K_V1.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


def VGG11(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG11_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG11_BN(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG11_BN_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG13(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG13_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG13_BN(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG13_BN_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG16(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG16_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG16_BN(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG16_BN_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG19(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG19_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def VGG19_BN(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        VGG19_BN_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
