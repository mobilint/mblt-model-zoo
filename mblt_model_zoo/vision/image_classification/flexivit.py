"""
FlexiViT model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class FlexiViT_Small_Set(ModelInfoSet):
    """Configuration set for FlexiViT Small models."""

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
                "size": 252,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [240, 240],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class FlexiViT_Base_Set(ModelInfoSet):
    """Configuration set for FlexiViT Base models."""

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
                "size": 252,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [240, 240],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class FlexiViT_Large_Set(ModelInfoSet):
    """Configuration set for FlexiViT Large models."""

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
                "size": 252,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [240, 240],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


def FlexiViT_Small(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a FlexiViT_Small model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        FlexiViT_Small_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def FlexiViT_Base(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a FlexiViT_Base model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        FlexiViT_Base_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def FlexiViT_Large(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs a FlexiViT_Large model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        FlexiViT_Large_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
