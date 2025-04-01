from mblt_model_zoo.vision.utils.types import ModelInfo, ModelInfoSet
from mblt_model_zoo.vision.wrapper import MBLT_Engine
from typing import Optional, Union, List, Any


class YOLOv8n_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )


class YOLOv8s_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )


class YOLOv8m_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )


class YOLOv8l_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )


class YOLOv8x_Set(ModelInfoSet):
    DEFAULT = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )


class YOLOv8n(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8n_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8n_Set. Available types: {YOLOv8n_Set.__dict__.keys()}"
        model_cfg = YOLOv8n_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8n_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8n_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8s(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8s_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8s_Set. Available types: {YOLOv8s_Set.__dict__.keys()}"
        model_cfg = YOLOv8s_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8s_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8s_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8m(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8m_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8m_Set. Available types: {YOLOv8m_Set.__dict__.keys()}"
        model_cfg = YOLOv8m_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8m_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8m_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8l(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8l_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8l_Set. Available types: {YOLOv8l_Set.__dict__.keys()}"
        model_cfg = YOLOv8l_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8l_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8l_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8x(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8x_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8x_Set. Available types: {YOLOv8x_Set.__dict__.keys()}"
        model_cfg = YOLOv8x_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8x_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8x_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
