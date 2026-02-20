from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class EfficientNet_V2_S_Set(ModelInfoSet):
    """Configuration set for EfficientNet_V2_S models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/EfficientNet_V2_S",
            "filename": "efficientnet_v2_s_IMAGENET1K_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 384,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_M_Set(ModelInfoSet):
    """Configuration set for EfficientNet_V2_M models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/EfficientNet_V2_M",
            "filename": "efficientnet_v2_m_IMAGENET1K_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 480,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [480, 480],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_L_Set(ModelInfoSet):
    """Configuration set for EfficientNet_V2_L models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/EfficientNet_V2_L",
            "filename": "efficientnet_v2_l_IMAGENET1K_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 480,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [480, 480],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_S(MBLT_Engine):
    """EfficientNet_V2_S model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the EfficientNet_V2_S engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            EfficientNet_V2_S_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class EfficientNet_V2_M(MBLT_Engine):
    """EfficientNet_V2_M model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the EfficientNet_V2_M engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            EfficientNet_V2_M_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class EfficientNet_V2_L(MBLT_Engine):
    """EfficientNet_V2_L model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the EfficientNet_V2_L engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            EfficientNet_V2_L_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
