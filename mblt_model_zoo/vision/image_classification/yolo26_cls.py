from collections import OrderedDict
from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLO26nCls_Set(ModelInfoSet):
    """Configuration set for YOLO26nCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLO26n-cls",
                "filename": "yolo26n-cls.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
                "Reader": {
                    "style": "pil",
                },
                "Resize": {
                    "size": 224,
                    "interpolation": "bilinear",
                },
                "CenterCrop": {
                    "size": [224, 224],
                },
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "cv",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )


class YOLO26sCls_Set(ModelInfoSet):
    """Configuration set for YOLO26sCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLO26s-cls",
                "filename": "yolo26s-cls.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
                "Reader": {
                    "style": "pil",
                },
                "Resize": {
                    "size": 224,
                    "interpolation": "bilinear",
                },
                "CenterCrop": {
                    "size": [224, 224],
                },
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "cv",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )


class YOLO26mCls_Set(ModelInfoSet):
    """Configuration set for YOLO26mCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLO26m-cls",
                "filename": "yolo26m-cls.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
                "Reader": {
                    "style": "pil",
                },
                "Resize": {
                    "size": 224,
                    "interpolation": "bilinear",
                },
                "CenterCrop": {
                    "size": [224, 224],
                },
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "cv",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )


class YOLO26lCls_Set(ModelInfoSet):
    """Configuration set for YOLO26lCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLO26l-cls",
                "filename": "yolo26l-cls.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
                "Reader": {
                    "style": "pil",
                },
                "Resize": {
                    "size": 224,
                    "interpolation": "bilinear",
                },
                "CenterCrop": {
                    "size": [224, 224],
                },
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "cv",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )


class YOLO26xCls_Set(ModelInfoSet):
    """Configuration set for YOLO26xCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLO26x-cls",
                "filename": "yolo26x-cls.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
                "Reader": {
                    "style": "pil",
                },
                "Resize": {
                    "size": 224,
                    "interpolation": "bilinear",
                },
                "CenterCrop": {
                    "size": [224, 224],
                },
                "SetOrder": {"shape": "HWC"},
                "Normalize": {
                    "style": "cv",
                },
            }
        ),
        post_cfg=OrderedDict({"task": "image_classification"}),
    )


class YOLO26nCls(MBLT_Engine):
    """YOLO26nCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26nCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26nCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26sCls(MBLT_Engine):
    """YOLO26sCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26sCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26sCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26mCls(MBLT_Engine):
    """YOLO26mCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26mCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26mCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26lCls(MBLT_Engine):
    """YOLO26lCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26lCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26lCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLO26xCls(MBLT_Engine):
    """YOLO26xCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLO26xCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLO26xCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
