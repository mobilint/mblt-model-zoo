from collections import OrderedDict
from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8nCls_Set(ModelInfoSet):
    """Configuration set for YOLOv8nCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLOv8n-cls",
                "filename": "yolov8n-cls.mxq",
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


class YOLOv8sCls_Set(ModelInfoSet):
    """Configuration set for YOLOv8sCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLOv8s-cls",
                "filename": "yolov8s-cls.mxq",
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


class YOLOv8mCls_Set(ModelInfoSet):
    """Configuration set for YOLOv8mCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLOv8m-cls",
                "filename": "yolov8m-cls.mxq",
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


class YOLOv8lCls_Set(ModelInfoSet):
    """Configuration set for YOLOv8lCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLOv8l-cls",
                "filename": "yolov8l-cls.mxq",
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


class YOLOv8xCls_Set(ModelInfoSet):
    """Configuration set for YOLOv8xCls models."""

    DEFAULT = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/YOLOv8x-cls",
                "filename": "yolov8x-cls.mxq",
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


class YOLOv8nCls(MBLT_Engine):
    """YOLOv8nCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8nCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8nCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8sCls(MBLT_Engine):
    """YOLOv8sCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8sCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8sCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8mCls(MBLT_Engine):
    """YOLOv8mCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8mCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8mCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8lCls(MBLT_Engine):
    """YOLOv8lCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8lCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8lCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8xCls(MBLT_Engine):
    """YOLOv8xCls model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8xCls engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """

        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8xCls_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
