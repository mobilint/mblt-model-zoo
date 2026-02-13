"""
YOLO26 model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO26n_Set(ModelInfoSet):
    """Configuration set for YOLO26n models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26n",
            "filename": "yolo26n.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "dflfree": True,  # dfl free yolo
        },
    )


class YOLO26s_Set(ModelInfoSet):
    """Configuration set for YOLO26s models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26s",
            "filename": "yolo26s.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "dflfree": True,  # dfl free yolo
        },
    )


class YOLO26m_Set(ModelInfoSet):
    """Configuration set for YOLO26m models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26m",
            "filename": "yolo26m.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "dflfree": True,  # dfl free yolo
        },
    )


class YOLO26l_Set(ModelInfoSet):
    """Configuration set for YOLO26l models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26l",
            "filename": "yolo26l.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "dflfree": True,  # dfl free yolo
        },
    )


class YOLO26x_Set(ModelInfoSet):
    """Configuration set for YOLO26x models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLO26x",
            "filename": "yolo26x.mxq",
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "dflfree": True,  # dfl free yolo
        },
    )


class YOLO26n(MBLT_Engine):
    """YOLO26n model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26n engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26n_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26s(MBLT_Engine):
    """YOLO26s model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26s engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26s_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26m(MBLT_Engine):
    """YOLO26m model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26m engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26m_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26l(MBLT_Engine):
    """YOLO26l model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26l engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26l_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26x(MBLT_Engine):
    """YOLO26x model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26x engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26x_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
