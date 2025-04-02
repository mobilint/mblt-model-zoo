from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv9t_Set(ModelInfoSet):
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


class YOLOv9s_Set(ModelInfoSet):
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


class YOLOv9m_Set(ModelInfoSet):
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


class YOLOv9c_Set(ModelInfoSet):
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


class YOLOv9e_Set(ModelInfoSet):
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


class YOLOv9t(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9t_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9t_Set. Available types: {YOLOv9t_Set.__dict__.keys()}"
        model_cfg = YOLOv9t_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9t_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9t_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9s(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9s_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9s_Set. Available types: {YOLOv9s_Set.__dict__.keys()}"
        model_cfg = YOLOv9s_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9s_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9s_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9m(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9m_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9m_Set. Available types: {YOLOv9m_Set.__dict__.keys()}"
        model_cfg = YOLOv9m_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9m_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9m_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9c(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9c_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9c_Set. Available types: {YOLOv9c_Set.__dict__.keys()}"
        model_cfg = YOLOv9c_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9c_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9c_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9e(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv9e_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv9e_Set. Available types: {YOLOv9e_Set.__dict__.keys()}"
        model_cfg = YOLOv9e_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv9e_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv9e_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
