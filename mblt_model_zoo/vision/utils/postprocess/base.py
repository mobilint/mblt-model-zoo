"""Base postprocessing classes."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .common import process_mask_upsample


class PostBase(ABC):
    """Abstract base class for post-processing operations."""

    def __init__(self):
        """Initialize the PostBase class."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x):
        """
        Apply post-processing to the input.

        Args:
            x: Raw model output.
        """

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
    """YOLO post-processing base class."""

    def __init__(self, pre_cfg: dict, post_cfg: dict):
        """
        Initialize the YOLOPostBase class.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
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
        self.conf_thres = 0.25
        self.inv_conf_thres = 1 / self.conf_thres
        self.iou_thres = 0.45

    def __call__(self, x, conf_thres=None, iou_thres=None):
        """
        Run the full YOLO post-processing pipeline.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model output.
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): NMS IoU threshold.

        Returns:
            list: List of detections for each image in the batch.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        x = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return x

    def set_threshold(self, conf_thres: float = None, iou_thres: float = None):
        """
        Set confidence and IoU thresholds.

        Args:
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold for NMS.

        Raises:
            AssertionError: If thresholds are invalid or missing.
        """
        assert (
            conf_thres is not None and iou_thres is not None
        ), "conf_thres and iou_thres should be provided in yolo_postprocess "
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.conf_thres = conf_thres
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)
        self.iou_thres = iou_thres

    def check_input(self, x: Union[TensorLike, ListTensorLike]):
        """
        Validate and prepare the input for post-processing.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model output.

        Returns:
            list: List of torch.Tensors on the correct device.
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

        return x

    @abstractmethod
    def rearrange(self, x):
        """
        Rearrange the input tensors for decoding.

        Args:
            x (ListTensorLike): Model output tensors.
        """

    @abstractmethod
    def decode(self, x):
        """
        Decode the model outputs into boxes and scores.

        Args:
            x (ListTensorLike): Rearranged output tensors.
        """

    @abstractmethod
    def nms(self, x):
        """
        Apply Non-Maximum Suppression (NMS) to the decoded detections.

        Args:
            x: Decoded detections.
        """

    def masking(self, x, proto_outs):
        """
        Generate and apply masks for instance segmentation.

        Args:
            x: Post-processed detections.
            proto_outs: Prototype outputs from the model.

        Returns:
            list: List of [detections, masks] for each image.
        """
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
