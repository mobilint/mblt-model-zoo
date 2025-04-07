from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch
from ..types import TensorLike, ListTensorLike
from .common import *


class PostBase(ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x):
        pass

    def to(self, device: Union[str, torch.device]):
        """Move the operations to the specified device.

        Args:
            device (Union[str, torch.device]): Device to move the operations to.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")

        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))


class YOLOPostBase(PostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__()
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

    def __call__(self, x, conf_thres=None, iou_thres=None):
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        x = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return x

    def set_threshold(self, conf_thres: float = None, iou_thres: float = None):
        assert (
            conf_thres is not None and iou_thres is not None
        ), "conf_thres and iou_thres should be provided in yolo_postprocess "
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.conf_thres = conf_thres
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)
        self.iou_thres = iou_thres

    def check_input(self, x: Union[TensorLike, ListTensorLike]):
        if isinstance(x, np.ndarray):
            x = [torch.from_numpy(x)]
        elif isinstance(x, torch.Tensor):
            x = [x]

        assert isinstance(x, list), f"Got unexpected type for x={type(x)}."

        if isinstance(x[0], np.ndarray):
            x = [torch.from_numpy(xi).to(self.device) for xi in x]
        elif isinstance(x[0], torch.Tensor):
            x = [xi.to(self.device) for xi in x]

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

    def masking(self, x, proto_outs):
        masks = []
        for pred, proto in zip(x, proto_outs):
            if len(pred) == 0:
                masks.append(
                    torch.zeros(
                        (0, self.imh, self.imw), dtype=torch.float32, device=self.device
                    )
                )
                continue
            masks.append(
                process_mask_upsample(
                    proto, pred[:, 6:], pred[:, :4], [self.imh, self.imw]
                )
            )
        return [[xi, mask] for xi, mask in zip(x, masks)]
