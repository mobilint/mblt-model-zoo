from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class VisFormer_Tiny_Set(ModelInfoSet):
    """Configuration set for VisFormer_Tiny models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/VisFormer_Tiny",
            "filename": "visformer_tiny.mxq",
            "revision": "main",
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


class VisFormer_Small_Set(ModelInfoSet):
    """Configuration set for VisFormer_Small models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/VisFormer_Small",
            "filename": "visformer_small.mxq",
            "revision": "main",
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


class VisFormer_Tiny(MBLT_Engine):
    """VisFormer_Tiny model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the VisFormer_Tiny engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

    model_cfg, pre_cfg, post_cfg = self._get_configs(
        VisFormer_Tiny_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
    super().__init__(model_cfg, pre_cfg, post_cfg)


class VisFormer_Small(MBLT_Engine):
    """VisFormer_Small model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the VisFormer_Small engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

    model_cfg, pre_cfg, post_cfg = self._get_configs(
        VisFormer_Small_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
    super().__init__(model_cfg, pre_cfg, post_cfg)
