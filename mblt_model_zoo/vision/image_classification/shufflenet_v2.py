"""
ShuffleNet V2 model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ShuffleNet_V2_X0_5_Set(ModelInfoSet):
    """Configuration set for ShuffleNet V2 x0.5 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ShuffleNet_V2_X0_5",
            "filename": "shufflenet_v2_x0_5_IMAGENET1K_V1.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ShuffleNet_V2_X1_0_Set(ModelInfoSet):
    """Configuration set for ShuffleNet V2 x1.0 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ShuffleNet_V2_X1_0",
            "filename": "shufflenet_v2_x1_0_IMAGENET1K_V1.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ShuffleNet_V2_X1_5_Set(ModelInfoSet):
    """Configuration set for ShuffleNet V2 x1.5 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ShuffleNet_V2_X1_5",
            "filename": "shufflenet_v2_x1_5_IMAGENET1K_V1.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ShuffleNet_V2_X2_0_Set(ModelInfoSet):
    """Configuration set for ShuffleNet V2 x2.0 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ShuffleNet_V2_X2_0",
            "filename": "shufflenet_v2_x2_0_IMAGENET1K_V1.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ShuffleNet_V2_X1_0(MBLT_Engine):
    """ShuffleNet_V2_X1_0 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ShuffleNet_V2_X1_0 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ShuffleNet_V2_X1_0_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ShuffleNet_V2_X1_5(MBLT_Engine):
    """ShuffleNet_V2_X1_5 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ShuffleNet_V2_X1_5 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ShuffleNet_V2_X1_5_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ShuffleNet_V2_X2_0(MBLT_Engine):
    """ShuffleNet_V2_X2_0 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ShuffleNet_V2_X2_0 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ShuffleNet_V2_X2_0_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
