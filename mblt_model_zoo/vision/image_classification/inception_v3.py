from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class Inception_V3_Set(ModelInfoSet):
    """Inception_V3 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/inception_v3_IMAGENET1K_V1/aries/single/inception_v3_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/inception_v3_IMAGENET1K_V1/aries/multi/inception_v3_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/inception_v3_IMAGENET1K_V1/aries/global/inception_v3_IMAGENET1K_V1.mxq",
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
            "Normalize": {"style": "torch"},
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
    Load the Inception_V3 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        Inception_V3_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
