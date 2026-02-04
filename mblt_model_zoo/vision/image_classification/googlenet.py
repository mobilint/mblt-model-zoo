"""
GoogLeNet model definition.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class GoogLeNet_Set(ModelInfoSet):
    """Configuration set for GoogLeNet models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/GoogLeNet",
            "filename": "GoogLeNet.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


def GoogLeNet(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a GoogLeNet model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        GoogLeNet_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
