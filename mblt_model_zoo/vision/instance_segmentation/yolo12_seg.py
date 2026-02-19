"""
YOLO12 Segmentation model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO12nSeg_Set(ModelInfoSet):
    """Configuration set for YOLO12nSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12n-seg",
            "filename": "yolo12n-seg.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )

    TURBO = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12n-seg",
            "filename": "yolo12n-seg.mxq",
            "revision": "TURBO",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLO12sSeg_Set(ModelInfoSet):
    """Configuration set for YOLO12sSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12s-seg",
            "filename": "yolo12s-seg.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )

    TURBO = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12s-seg",
            "filename": "yolo12s-seg.mxq",
            "revision": "TURBO",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLO12mSeg_Set(ModelInfoSet):
    """Configuration set for YOLO12mSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12m-seg",
            "filename": "yolo12m-seg.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )

    TURBO = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12m-seg",
            "filename": "yolo12m-seg.mxq",
            "revision": "TURBO",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLO12lSeg_Set(ModelInfoSet):
    """Configuration set for YOLO12lSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12l-seg",
            "filename": "yolo12l-seg.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )

    TURBO = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12l-seg",
            "filename": "yolo12l-seg.mxq",
            "revision": "TURBO",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLO12xSeg_Set(ModelInfoSet):
    """Configuration set for YOLO12xSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12x-seg",
            "filename": "yolo12x-seg.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )

    TURBO = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO12x-seg",
            "filename": "yolo12x-seg.mxq",
            "revision": "TURBO",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLO12nSeg(MBLT_Engine):
    """YOLO12nSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO12nSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO12nSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO12sSeg(MBLT_Engine):
    """YOLO12sSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO12sSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO12sSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO12mSeg(MBLT_Engine):
    """YOLO12mSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO12mSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO12mSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO12lSeg(MBLT_Engine):
    """YOLO12lSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO12lSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO12lSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO12xSeg(MBLT_Engine):
    """YOLO12xSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO12xSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO12xSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
