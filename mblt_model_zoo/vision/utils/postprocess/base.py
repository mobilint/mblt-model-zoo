from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .common import nmsout2eval, process_mask_upsample


class PostBase(ABC):
    """Abstract base class for postprocessing."""

    def __init__(self):
        """Initialize PostBase."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x: Union[TensorLike, ListTensorLike], *args: Any, **kwargs: Any) -> Any:
        """Executes postprocessing on the model output.

        Args:
            x (Union[TensorLike, ListTensorLike]): Input tensor or list of tensors from the model.
            *args (Any): Additional positional arguments depending on the specific task.
            **kwargs (Any): Additional keyword arguments depending on the specific task.

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
        letterbox_cfg = pre_cfg.get("LetterBox")
        if letterbox_cfg is None:
            raise ValueError("LetterBox configuration should be provided in pre_cfg")
        img_size = letterbox_cfg["img_size"]
        self.imh: int
        self.imw: int
        if isinstance(img_size, int):
            self.imh = self.imw = img_size
        elif isinstance(img_size, list):
            assert len(img_size) == 2, "img_size should be a list of two integers"
            self.imh, self.imw = img_size
        nc = post_cfg.get("nc")
        if nc is None:
            raise ValueError("nc should be provided in post_cfg")
        self.nc: int = nc
        self.anchors: Optional[Union[list, torch.Tensor]] = post_cfg.get("anchors", None)  # anchor coordinates
        self.stride: list[int] | torch.Tensor
        self.nl: int
        self.na: int
        if self.anchors is None:
            nl = post_cfg.get("nl")
            assert nl is not None, "nl should be provided in post_cfg"
            self.nl = nl
            if self.nl == 2:
                self.stride = [2 ** (4 + i) for i in range(self.nl)]
            else:
                self.stride = [2 ** (3 + i) for i in range(self.nl)]
            self.make_anchors()
        else:
            assert isinstance(self.anchors, list), "anchors should be a list"
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
        self.n_extra: int = post_cfg.get("n_extra", 0)
        task = post_cfg.get("task")
        if task is None:
            raise ValueError("task should be provided in post_cfg")
        self.task: str = task

    def anchors_as_list(self) -> list:
        """Return anchors as the configured anchor list."""
        if not isinstance(self.anchors, list):
            raise TypeError("anchors should be a list for anchor-based YOLO postprocessing.")
        return self.anchors

    def anchors_as_tensor(self) -> torch.Tensor:
        """Return anchors as the generated anchor-point tensor."""
        if not isinstance(self.anchors, torch.Tensor):
            raise TypeError("anchors should be a tensor for anchor-free YOLO postprocessing.")
        return cast(torch.Tensor, self.anchors)

    def stride_as_tensor(self) -> torch.Tensor:
        """Return strides as the generated stride tensor."""
        if not isinstance(self.stride, torch.Tensor):
            raise TypeError("stride should be a tensor for anchor-free YOLO postprocessing.")
        return cast(torch.Tensor, self.stride)

    def __call__(self, x: Union[TensorLike, ListTensorLike], conf_thres: float, iou_thres: float):
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
        checked_input = self.check_input(x)

        predictions, proto_outs = self._pre_process(checked_input)

        nms_output = self.nms(predictions)

        if proto_outs is not None:
            return self.masking(nms_output, proto_outs)
        return nms_output

    def _pre_process(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Protected method to preprocess inputs into (predictions, prototypes).

        Args:
            x: List of input tensors.

        Returns:
            Tuple of (predictions, prototypes). Prototypes may be None.
        """
        if len(x) == 1:
            converted = self.conversion(x)
            if not isinstance(converted, torch.Tensor):
                raise TypeError("conversion should return a tensor for single-output YOLO postprocessing.")
            return self.filter_conversion(converted), None
        rearranged = self.rearrange(x)
        return self.decode(rearranged), None

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple,
        img0_shape: Union[tuple, List[tuple]],
    ) -> tuple:
        """Converts NMS output to evaluation format (labels, boxes, scores).

        Args:
            nms_out: NMS output (tensor or list of tensors).
            img1_shape: Resized image shape (height, width).
            img0_shape: Original image shape(s).

        Returns:
            Tuple: task-specific results.
                - Detection: (labels_list, boxes_list, scores_list)
                - Segmentation/Pose: (labels_list, boxes_list, scores_list, extra_list)
        """

        return nmsout2eval(nms_out, img1_shape, img0_shape)

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
            stride_tensor.append(torch.full((ny * nx, 1), strd, dtype=torch.float32, device=self.device))
        self.anchors = torch.cat(anchor_points, dim=0).permute(1, 0)
        self.stride = torch.cat(stride_tensor, dim=0).permute(1, 0)

    def set_threshold(self, conf_thres: Optional[float] = None, iou_thres: Optional[float] = None) -> None:
        """Set confidence and IoU thresholds.
        Args:
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.
        """
        assert conf_thres is not None and iou_thres is not None, (
            "conf_thres and iou_thres should be provided in yolo_postprocess "
        )
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)

    def check_input(self, x: Union[TensorLike, ListTensorLike]) -> List[torch.Tensor]:
        """Check and prepare input tensors.
        Args:
            x (Union[TensorLike, ListTensorLike]): Input tensor or list of tensors.
        Returns:
            list[torch.Tensor]: List of tensors on the correct device.
        """
        if isinstance(x, np.ndarray):
            tensors = [torch.from_numpy(x)]
        elif isinstance(x, torch.Tensor):
            tensors = [x]
        else:
            assert isinstance(x, list), f"Got unexpected type for x={type(x)}."
            if isinstance(x[0], np.ndarray):
                tensors = [torch.from_numpy(xi).to(self.device) for xi in x]
            elif isinstance(x[0], torch.Tensor):
                tensors = [xi.to(self.device) for xi in x]
            else:
                raise TypeError(f"Got unexpected element type for x[0]={type(x[0])}.")
        return self.check_dim(tensors)

    def check_dim(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
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
    def rearrange(self, x: List[torch.Tensor]) -> Any:
        """Rearranges raw model outputs into a task-specific intermediate form.

        Args:
            x: Raw output tensors from the model.

        Returns:
            A task-specific intermediate representation used by ``decode``.
        """

    @abstractmethod
    def decode(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decodes rearranged outputs into per-image detection tensors.

        Args:
            x: Rearranged output tensors.

        Returns:
            Decoded detections for each image in the batch.
        """

    def conversion(self, x: List[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Converts raw outputs into a task-specific intermediate form.

        Args:
            x: Input tensors.

        Returns:
            A converted detection tensor, or a ``(detections, prototypes)`` tuple
            for segmentation-style subclasses.
        """
        assert len(x) == 1, f"Assume return is a single output, but got {len(x)} outputs"
        return x[0]

    @abstractmethod
    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Filters converted outputs into per-image detections before NMS.

        Args:
            x: Converted output tensor.

        Returns:
            Filtered detections for each image in the batch.
        """

    @abstractmethod
    def nms(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Performs non-maximum suppression on decoded detections.

        Args:
            x: Decoded detections for each image.

        Returns:
            Detections after NMS for each image in the batch.
        """

    def masking(self, x: List[torch.Tensor], proto_outs: List[torch.Tensor]):
        """Apply masking to detection results.
        Args:
            x: Detection results.
            proto_outs: Prototype outputs for masks.
        Returns:
            list: Detection results with masks.
        """
        masks = []
        for pred, proto in zip(x, proto_outs):
            proto = proto.permute(2, 0, 1)  # [mask_h, mask_w, mask_dim] -> [mask_dim, mask_h, mask_w]
            if len(pred) == 0:
                masks.append(torch.zeros((0, self.imh, self.imw), dtype=torch.float32, device=self.device))
                continue
            masks.append(process_mask_upsample(proto, pred[:, 6:], pred[:, :4], [self.imh, self.imw]))
        return [[xi, mask] for xi, mask in zip(x, masks)]
