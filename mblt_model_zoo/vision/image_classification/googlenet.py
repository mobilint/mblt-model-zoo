"""
GoogLeNet model definition.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class GoogLeNet_Set(ModelInfoSet):
    """Configuration set for GoogLeNet models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/GoogLeNet",
            "filename": "googlenet_IMAGENET1K_V1.mxq",
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


class GoogLeNet(MBLT_Engine):
    """GoogLeNet model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the GoogLeNet engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            GoogLeNet_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
