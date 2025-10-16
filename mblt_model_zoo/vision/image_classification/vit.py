from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ViT_Tiny_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_224/aries/single/vit_tiny_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_224/aries/multi/vit_tiny_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_224/aries/global/vit_tiny_patch16_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Tiny_Patch16_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_384/aries/single/vit_tiny_patch16_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_384/aries/multi/vit_tiny_patch16_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_tiny_patch16_384/aries/global/vit_tiny_patch16_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Small_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_224/aries/single/vit_small_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_224/aries/multi/vit_small_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_224/aries/global/vit_small_patch16_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Small_Patch16_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_384/aries/single/vit_small_patch16_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_384/aries/multi/vit_small_patch16_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch16_384/aries/global/vit_small_patch16_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Small_Patch32_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_224/aries/single/vit_small_patch32_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_224/aries/multi/vit_small_patch32_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_224/aries/global/vit_small_patch32_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Small_Patch32_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_384/aries/single/vit_small_patch32_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_384/aries/multi/vit_small_patch32_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_small_patch32_384/aries/global/vit_small_patch32_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Base_Patch8_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch8_224/aries/single/vit_base_patch8_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch8_224/aries/multi/vit_base_patch8_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch8_224/aries/global/vit_base_patch8_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Base_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_224/aries/single/vit_base_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_224/aries/multi/vit_base_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_224/aries/global/vit_base_patch16_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Base_Patch16_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_384/aries/single/vit_base_patch16_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_384/aries/multi/vit_base_patch16_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch16_384/aries/global/vit_base_patch16_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Base_Patch32_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_224/aries/single/vit_base_patch32_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_224/aries/multi/vit_base_patch32_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_224/aries/global/vit_base_patch32_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Base_Patch32_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_384/aries/single/vit_base_patch32_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_384/aries/multi/vit_base_patch32_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_base_patch32_384/aries/global/vit_base_patch32_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Large_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_224/aries/single/vit_large_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_224/aries/multi/vit_large_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_224/aries/global/vit_large_patch16_224.mxq",
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
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Large_Patch16_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_384/aries/single/vit_large_patch16_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_384/aries/multi/vit_large_patch16_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch16_384/aries/global/vit_large_patch16_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class ViT_Large_Patch32_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch32_384/aries/single/vit_large_patch32_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch32_384/aries/multi/vit_large_patch32_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/vit_large_patch32_384/aries/global/vit_large_patch32_384.mxq",
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
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


def ViT_Tiny_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Tiny_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Tiny_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Tiny_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Small_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Small_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Small_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Small_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Small_Patch32_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Small_Patch32_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Small_Patch32_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Small_Patch32_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Base_Patch8_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Base_Patch8_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Base_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Base_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Base_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Base_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Base_Patch32_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Base_Patch32_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Base_Patch32_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Base_Patch32_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Large_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Large_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Large_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Large_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ViT_Large_Patch32_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ViT_Large_Patch32_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
