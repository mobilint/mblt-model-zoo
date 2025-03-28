from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch
from mblt_models.vision.utils.types import TensorLike, ListTensorLike


class PostBase(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class YOLOPostBase(PostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)

        img_size = pre_cfg.get("YoloPre")["img_size"]

        if isinstance(img_size, int):
            self.imh = self.imw = img_size
        elif isinstance(img_size, list):
            self.imh, self.imw = img_size

        self.nc = post_cfg.get("nc")
        self.anchors = post_cfg.get("anchors", None)  # anchor coordinates
        if self.anchors is None:
            self.anchorless = True
            self.nl = post_cfg.get("nl")
            assert self.nl is not None, "nl should be provided for anchorless model"
        else:
            self.anchorless = False
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2

        self.n_extra = post_cfg.get("n_extra", 0)
        self.task = post_cfg.get("task")

    def __call__(self, x):
        x = self.check_input(x)
        x = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return x

    def check_input(self, x: Union[TensorLike, ListTensorLike]):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            x = [x]
        assert isinstance(x, list), f"Got unexpected type for x={type(x)}."
        if isinstance(x[0], np.ndarray):
            x = [torch.from_numpy(xx) for xx in x]
        assert all(
            isinstance(xx, torch.Tensor) for xx in x
        ), f"Got unexpected type for x={type(x)}."

        return x

    @abstractmethod
    def rearrange(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def nms(self, x):
        pass
