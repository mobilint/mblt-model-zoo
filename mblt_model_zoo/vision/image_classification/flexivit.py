from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class FlexiViT_Small_Set(ModelInfoSet):
    """FlexiViT_Small model info set."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/flexivit_small/aries/single/flexivit_small.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/flexivit_small/aries/multi/flexivit_small.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/flexivit_small/aries/global/flexivit_small.mxq",
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
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class FlexiViT_Base_Set(ModelInfoSet):
    """FlexiViT_Base model info set."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/flexivit_base/aries/single/flexivit_base.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/flexivit_base/aries/multi/flexivit_base.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/flexivit_base/aries/global/flexivit_base.mxq",
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
            "Normalize": {"style": "tf"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class FlexiViT_Large_Set(ModelInfoSet):
    """FlexiViT_Large model info set."""

    DEFAULT = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/flexivit_large/aries/single/flexivit_large.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/flexivit_large/aries/multi/flexivit_large.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/flexivit_large/aries/global/flexivit_large.mxq",
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
            "Normalize": {"style": "tf"},
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
    Load the FlexiViT_Small model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
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
    Load the FlexiViT_Base model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
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
    Load the FlexiViT_Large model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        FlexiViT_Large_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
