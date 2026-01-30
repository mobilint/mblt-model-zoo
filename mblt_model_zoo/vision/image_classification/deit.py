"""
DeiT (Data-efficient Image Transformers) model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class DeiT_Tiny_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Tiny Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Small_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Small Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Base_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Base Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT_Base_Patch16_384_Set(ModelInfoSet):
    """Configuration set for DeiT Base Patch16 384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


def DeiT_Tiny_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a DeiT_Tiny_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
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
    """
    Constructs a DeiT_Small_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
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
    """
    Constructs a DeiT_Base_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
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
    """
    Constructs a DeiT_Base_Patch16_384 model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT_Base_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
