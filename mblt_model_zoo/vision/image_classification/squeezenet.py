"""
SqueezeNet model definitions.
"""

from collections import OrderedDict
from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class SqueezeNet1_0_Set(ModelInfoSet):
    """Configuration set for SqueezeNet1_0 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/SqueezeNet1_0",
                "filename": "squeezenet1_0_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class SqueezeNet1_1_Set(ModelInfoSet):
    """Configuration set for SqueezeNet1_1 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/SqueezeNet1_1",
                "filename": "squeezenet1_1_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class SqueezeNet1_0(MBLT_Engine):
    """SqueezeNet1_0 model."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the SqueezeNet1_0 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            SqueezeNet1_0_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class SqueezeNet1_1(MBLT_Engine):
    """SqueezeNet1_1 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the SqueezeNet1_1 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            SqueezeNet1_1_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
