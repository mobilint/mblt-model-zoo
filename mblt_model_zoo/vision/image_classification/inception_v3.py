"""
Inception V3 model definition.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class Inception_V3_Set(ModelInfoSet):
    """Configuration set for Inception V3 models."""

    IMAGENET1K_V1 = ModelInfo(
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
                "size": 342,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [299, 299],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


def Inception_V3(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Constructs an Inception_V3 model engine.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Type of the model to use. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode. Defaults to "global".
        product (str, optional): Target product. Defaults to "aries".

    Returns:
        MBLT_Engine: The constructed model engine.
    """
    return MBLT_Engine.from_model_info_set(
        Inception_V3_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
