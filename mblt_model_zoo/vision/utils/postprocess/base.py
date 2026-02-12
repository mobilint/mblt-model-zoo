from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .common import process_mask_upsample


class PostBase(ABC):
    """Abstract base class for postprocessing."""

    def __init__(self):
        """Initialize PostBase."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x: Union[TensorLike, ListTensorLike]):
        """Executes postprocessing on the model output.

        Args:
            x (Union[TensorLike, ListTensorLike]): Input tensor or list of tensors from the model.

        Returns:
            Any: Postprocessed results, format depends on the specific task.
        """
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
    """Base class for YOLO postprocessing."""

    def __init__(self, pre_cfg: dict, post_cfg: dict):
        """Initializes the YOLOPostBase.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        super().__init__()
        img_size = pre_cfg.get("YoloPre")["img_size"]
        if isinstance(img_size, int):
            self.imh = self.imw = img_size
        elif isinstance(img_size, list):
            assert len(img_size) == 2, "img_size should be a list of two integers"
            self.imh, self.imw = img_size
        self.nc = post_cfg.get("nc")
        assert self.nc is not None, "nc should be provided in post_cfg"
        self.anchors = post_cfg.get("anchors", None)  # anchor coordinates
        if self.anchors is None:
            self.nl = post_cfg.get("nl")
            assert self.nl is not None, "nl should be provided in post_cfg"
            if self.nl == 2:
                self.stride = [2 ** (4 + i) for i in range(self.nl)]
            else:
                self.stride = [2 ** (3 + i) for i in range(self.nl)]
            self.make_anchors()
        else:
            assert isinstance(self.anchors, list), "anchors should be a list"
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
        self.n_extra = post_cfg.get("n_extra", 0)
        self.task = post_cfg.get("task")

    def __call__(
        self, x: Union[TensorLike, ListTensorLike], conf_thres: float, iou_thres: float
    ):
        """Executes YOLO postprocessing.

        Includes rearranging, decoding, and NMS.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model outputs.
            conf_thres (float): Confidence threshold for detection.
            iou_thres (float): IoU threshold for NMS.

        Returns:
            list: List of detections per image.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        if len(x) == 1:
            x = self.conversion(x)
            x = self.filter_conversion(x)
        else:
            x = self.rearrange(x)
            x = self.decode(x)
        x = self.nms(x)
        return x

    def make_anchors(self, offset=0.5):
        """
        Generate anchor points and stride tensors based on image size and strides.
        Args:
            offset (float, optional): Offset for anchor points. Defaults to 0.5.
        """
        anchor_points, stride_tensor = [], []
        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]
        for strd in strides:
            ny, nx = self.imh // strd, self.imw // strd
            sy = torch.arange(ny, dtype=torch.float32, device=self.device) + offset
            sx = torch.arange(nx, dtype=torch.float32, device=self.device) + offset
            yv, xv = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
            stride_tensor.append(
                torch.full((ny * nx, 1), strd, dtype=torch.float32, device=self.device)
            )
        self.anchors = torch.cat(anchor_points, dim=0).permute(1, 0)
        self.stride = torch.cat(stride_tensor, dim=0).permute(1, 0)

    def set_threshold(self, conf_thres: float = None, iou_thres: float = None):
        """Set confidence and IoU thresholds.
        Args:
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.
        """
        assert (
            conf_thres is not None and iou_thres is not None
        ), "conf_thres and iou_thres should be provided in yolo_postprocess "
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)

    def check_input(self, x: Union[TensorLike, ListTensorLike]):
        """Check and prepare input tensors.
        Args:
            x (Union[TensorLike, ListTensorLike]): Input tensor or list of tensors.
        Returns:
            list[torch.Tensor]: List of tensors on the correct device.
        """
        if isinstance(x, np.ndarray):
            x = [torch.from_numpy(x)]
        elif isinstance(x, torch.Tensor):
            x = [x]
        assert isinstance(x, list), f"Got unexpected type for x={type(x)}."
        if isinstance(x[0], np.ndarray):
            x = [torch.from_numpy(xi).to(self.device) for xi in x]
        elif isinstance(x[0], torch.Tensor):
            x = [xi.to(self.device) for xi in x]
        return self.check_dim(x)

    def check_dim(self, x: List[torch.Tensor]):
        """Check tensor dimensions.
        Args:
            x (List[torch.Tensor]): List of tensors.
        Returns:
            List[torch.Tensor]: List of tensors with corrected dimensions.
        """
        y = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            elif xi.ndim == 4:
                pass
            else:
                raise ValueError(f"Got unexpected dim for xi={xi.ndim}.")
            y.append(xi)
        return y

    @abstractmethod
    def rearrange(self, x):
        """Rearrange the output."""
        """
        Rearrange the raw output tensors into a structured format.
        Args:
            x (list[torch.Tensor]): Raw output tensors from the model.
        Returns:
            torch.Tensor: Rearranged tensor.
        """

    @abstractmethod
    def decode(self, x):
        """Decode the output."""
        """
        Decode the model outputs into box coordinates and class scores.
        Args:
            x (torch.Tensor): Rearranged output tensor.
        Returns:
            torch.Tensor: Decoded tensor.
        """

    def conversion(self, x: List[torch.Tensor]):
        """Convert input tensors.
        Args:
            x (List[torch.Tensor]): Input tensors.
        Returns:
            torch.Tensor: Converted tensor.
        """
        assert (
            len(x) == 1
        ), f"Assume return is a single output, but got {len(x)} outputs"
        return x[0]

    @abstractmethod
    def filter_conversion(self, x):
        """Filter and convert outputs before NMS.
        Args:
            x: Input tensor.
        """

    @abstractmethod
    def nms(self, x):
        """Perform Non-Maximum Suppression.
        Perform Non-Maximum Suppression (NMS) on the detections.
        Args:
            x: Input tensor.
            x (torch.Tensor): Decoded detections.
        Returns:
            list[torch.Tensor]: Detections after NMS.
        """

    def masking(self, x, proto_outs):
        """Apply masking to detection results.
        Args:
            x: Detection results.
            proto_outs: Prototype outputs for masks.
        Returns:
            list: Detection results with masks.
        """
        masks = []
        for pred, proto in zip(x, proto_outs):
            proto = proto.permute(
                2, 0, 1
            )  # [mask_h, mask_w, mask_dim] -> [mask_dim, mask_h, mask_w]
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
