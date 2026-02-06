"""
FlexiViT model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class FlexiViT_Small_Set(ModelInfoSet):
    """Configuration set for FlexiViT Small models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/FlexiVit-Small",
            "filename": "flexivit-small.mxq",
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
            "repo_id": "mobilint/FlexiVit-Base",
            "filename": "flexivit-base.mxq",
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
            "repo_id": "mobilint/FlexiVit-Large",
            "filename": "flexivit-large.mxq",
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


class FlexiViT_Small(MBLT_Engine):
    """FlexiViT_Small model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the FlexiViT_Small engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            FlexiViT_Small_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class FlexiViT_Base(MBLT_Engine):
    """FlexiViT_Base model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the FlexiViT_Base engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            FlexiViT_Base_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class FlexiViT_Large(MBLT_Engine):
    """FlexiViT_Large model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the FlexiViT_Large engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            FlexiViT_Large_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
