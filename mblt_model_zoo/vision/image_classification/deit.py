from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class DeiT_Tiny_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/deit_tiny_patch16_224/aries/single/deit_tiny_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/deit_tiny_patch16_224/aries/multi/deit_tiny_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/deit_tiny_patch16_224/aries/global/deit_tiny_patch16_224.mxq",
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
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Small_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/deit_small_patch16_224/aries/single/deit_small_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/deit_small_patch16_224/aries/multi/deit_small_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/deit_small_patch16_224/aries/global/deit_small_patch16_224.mxq",
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
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Base_Patch16_224_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_224/aries/single/deit_base_patch16_224.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_224/aries/multi/deit_base_patch16_224.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_224/aries/global/deit_base_patch16_224.mxq",
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
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Base_Patch16_384_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_384/aries/single/deit_base_patch16_384.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_384/aries/multi/deit_base_patch16_384.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/deit_base_patch16_384/aries/global/deit_base_patch16_384.mxq",
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
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )


def DeiT_Tiny_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        DeiT_Tiny_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT_Small_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        DeiT_Small_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT_Base_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        DeiT_Base_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT_Base_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        DeiT_Base_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
