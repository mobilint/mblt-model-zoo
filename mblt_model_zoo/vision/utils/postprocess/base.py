from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .common import RatioPad, nmsout2eval, process_mask_upsample


class PostBase(ABC):
    """Abstract base class for postprocessing."""

    def __init__(self) -> None:
        """Initialize PostBase."""
        super().__init__()
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x: TensorLike | ListTensorLike, *args: Any, **kwargs: Any) -> Any:
        """Executes postprocessing on the model output.

        Args:
            x (TensorLike | ListTensorLike): Input tensor or list of tensors from the model.
            *args (Any): Additional positional arguments depending on the specific task.
            **kwargs (Any): Additional keyword arguments depending on the specific task.

        Returns:
            Any: Postprocessed results, format depends on the specific task.
        """
        pass

    def to(self, device: str | torch.device) -> None:
        """Move the operations to the specified device.
        Args:
            device (str | torch.device): Device to move the operations to.
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


class YOLODetectionPostBase(PostBase):
    """Base class for YOLO postprocessing."""

    NC_BY_DATASET_TASK: dict[tuple[str, str], int] = {
        ("coco", "object_detection"): 80,
        ("coco", "instance_segmentation"): 80,
        ("coco", "pose_estimation"): 1,
        ("dotav1", "obb"): 15,
        ("open-images-v7", "object_detection"): 601,
        ("widerface", "face_detection"): 1,
    }
    DEFAULT_NC_BY_TASK: dict[str, int] = {
        "object_detection": 80,
        "instance_segmentation": 80,
        "pose_estimation": 1,
        "obb": 15,
        "face_detection": 1,
    }

    def __init__(self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any], **kwargs) -> None:
        """Initialize the common YOLO detection postprocessor.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.

        Raises:
            TypeError: If unsupported keyword overrides are provided.
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
        task = post_cfg.get("task")
        if task is None:
            raise ValueError("task should be provided in post_cfg")
        self.task: str = task
        task_key = self.task.lower()
        dataset = post_cfg.get("dataset")
        self.dataset = dataset.lower() if isinstance(dataset, str) else None
        dataset_nc = self.NC_BY_DATASET_TASK.get((self.dataset, task_key)) if self.dataset is not None else None
        configured_nc = kwargs.pop("nc", post_cfg.get("nc"))
        if configured_nc is not None and dataset_nc is not None and int(configured_nc) != dataset_nc:
            raise ValueError(
                f"nc={configured_nc} conflicts with dataset '{self.dataset}' and task '{self.task}', "
                f"which require nc={dataset_nc}."
            )
        default_nc = dataset_nc if dataset_nc is not None else self.DEFAULT_NC_BY_TASK.get(task_key)
        nc = configured_nc if configured_nc is not None else default_nc
        if nc is None:
            raise ValueError(f"nc should be provided in post_cfg or kwargs for task '{self.task}'.")
        self.nc: int = int(nc)
        self.anchors: list[Any] | torch.Tensor | None = post_cfg.get("anchors", None)  # anchor coordinates
        self.stride: list[int] | torch.Tensor
        self.nl: int
        self.na: int
        self.conf_thres: float
        self.iou_thres: float
        self.inv_conf_thres: float

        self.e2e = bool(kwargs.pop("e2e", post_cfg.get("e2e", True)))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected YOLO postprocess kwargs: {unexpected}")

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
        self.conf_thres = float(post_cfg.get("conf_thres", 0.25))
        self.iou_thres = float(post_cfg.get("iou_thres", 0.7))
        self.set_threshold()

    def anchors_as_list(self) -> list[Any]:
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

    def __call__(
        self,
        x: TensorLike | ListTensorLike,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> list[Any]:
        """Executes YOLO postprocessing.

        Includes rearranging, decoding, and NMS.

        Args:
            x (TensorLike | ListTensorLike): Raw model outputs.
            conf_thres (float | None): Confidence threshold for detection.
            iou_thres (float | None): IoU threshold for NMS.

        Returns:
            list: List of detections per image.
        """
        self.set_threshold(conf_thres, iou_thres)
        final_detections, proto_outs = self.extract_final_outputs(x)
        if final_detections is not None:
            if proto_outs is not None:
                return self.masking(final_detections, proto_outs)
            return final_detections
        checked_input = self.check_input(x)

        if not self.e2e:
            return self.non_e2e(checked_input)

        predictions, proto_outs = self._pre_process(checked_input)

        nms_output = self.nms(predictions)

        if proto_outs is not None:
            return self.masking(nms_output, proto_outs)
        return nms_output

    def non_e2e(self, x: list[torch.Tensor]) -> Any:
        """Return the export-style postprocess output when end-to-end mode is disabled.

        Args:
            x: Checked raw model outputs.

        Returns:
            Export-style tensors whose batch dimensions remain intact.
        """
        if len(x) == 1:
            return self.conversion(x)
        return self.rearrange(x)

    def _pre_process(
        self,
        x: list[torch.Tensor],
    ) -> tuple[Any, torch.Tensor | list[torch.Tensor] | None]:
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
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
        ratio_pad: RatioPad | list[RatioPad | None] | None = None,
    ) -> tuple[Any, ...]:
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

        img0_shapes = [img0_shape] if isinstance(img0_shape, tuple) else img0_shape
        ratio_pads: list[RatioPad | None] | None
        if ratio_pad is None:
            ratio_pads = None
        elif isinstance(ratio_pad, list):
            ratio_pads = list(ratio_pad)
        else:
            ratio_pads = [ratio_pad]
        return nmsout2eval(nms_out, img1_shape, img0_shapes, ratio_pads=ratio_pads)

    def extract_final_outputs(
        self,
        x: TensorLike | ListTensorLike,
    ) -> tuple[list[torch.Tensor] | None, torch.Tensor | None]:
        """Extract already-decoded ONNX-style detections when present.

        Args:
            x: Raw postprocess input.

        Returns:
            A tuple of ``(detections, prototypes)`` when the input already contains
            final detections, otherwise ``(None, None)``.
        """

        final_det_dim = 6 + self.n_extra

        if isinstance(x, Sequence):
            if not x:
                return None, None

            normalized_detections: np.ndarray | torch.Tensor | None = None
            normalized_proto: torch.Tensor | None = None
            for output in x:
                if not isinstance(output, (np.ndarray, torch.Tensor)):
                    continue
                if normalized_detections is None:
                    normalized_detections = self._normalize_final_detection_tensor(output, final_det_dim)
                    if normalized_detections is not None:
                        continue
                if normalized_proto is None:
                    try:
                        normalized_proto = self._normalize_proto_batch(output)
                    except ValueError:
                        continue

            if normalized_detections is not None:
                return self._final_detection_batches(normalized_detections), normalized_proto
            return None, None

        normalized_x = self._normalize_final_detection_tensor(x, final_det_dim)
        if normalized_x is not None:
            return self._final_detection_batches(normalized_x), None

        return None, None

    def _normalize_final_detection_tensor(
        self,
        x: TensorLike,
        final_det_dim: int,
    ) -> np.ndarray | torch.Tensor | None:
        """Return a batched final-detection tensor when ``x`` already contains decoded rows."""

        while x.ndim == 4 and 1 in (x.shape[0], x.shape[1]):
            if x.shape[1] == 1:
                x = x[:, 0]
            elif x.shape[0] == 1:
                x = x[0]
        if x.ndim == 2 and x.shape[-1] == final_det_dim:
            x = x[None]
        if x.ndim == 3 and x.shape[-1] == final_det_dim:
            return x
        if x.ndim == 3 and x.shape[1] == final_det_dim:
            if isinstance(x, np.ndarray):
                return np.swapaxes(x, 1, 2)
            return x.transpose(1, 2)

        return None

    def _final_detection_batches(self, x: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Convert batched final detections to the internal per-image tensor list."""

        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).to(self.device)
        else:
            tensor = x.to(self.device)
        return [batch[batch[:, 4] > self.conf_thres] for batch in tensor]

    def _normalize_proto_batch(self, proto_outs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize prototype masks to ``(B, H, W, C)`` layout."""

        if isinstance(proto_outs, np.ndarray):
            proto = torch.from_numpy(proto_outs).to(self.device)
        else:
            proto = proto_outs.to(self.device)

        if proto.ndim != 4:
            raise ValueError(f"Expected 4D prototype tensor, got shape {tuple(proto.shape)}.")
        if proto.shape[-1] == self.n_extra:
            return proto
        if proto.shape[1] == self.n_extra:
            return proto.permute(0, 2, 3, 1)
        raise ValueError(f"Unsupported prototype tensor shape {tuple(proto.shape)}.")

    def make_anchors(self, offset: float = 0.5) -> None:
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

    def set_threshold(self, conf_thres: float | None = None, iou_thres: float | None = None) -> None:
        """Set confidence and IoU thresholds.
        Args:
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.
        """
        conf_thres = self.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.iou_thres if iou_thres is None else iou_thres
        assert 0 < conf_thres < 1, "conf_thres should be in (0, 1)"
        assert 0 < iou_thres < 1, "iou_thres should be in (0, 1)"
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.inv_conf_thres = -np.log(1 / conf_thres - 1)

    def check_input(self, x: TensorLike | ListTensorLike) -> list[torch.Tensor]:
        """Check and prepare input tensors.
        Args:
            x (TensorLike | ListTensorLike): Input tensor or list of tensors.
        Returns:
            list[torch.Tensor]: List of tensors on the correct device.
        """
        if isinstance(x, np.ndarray):
            tensors = [torch.from_numpy(x).to(self.device)]
        elif isinstance(x, torch.Tensor):
            tensor_input = cast(torch.Tensor, x)
            tensors = [tensor_input.to(self.device)]
        else:
            if not isinstance(x, Sequence):
                raise TypeError(f"Got unexpected type for x={type(x)}.")
            if all(isinstance(xi, np.ndarray) for xi in x):
                tensors = [torch.from_numpy(xi).to(self.device) for xi in x]
            elif all(isinstance(xi, torch.Tensor) for xi in x):
                torch_inputs = cast(Sequence[torch.Tensor], x)
                tensors = [xi.to(self.device) for xi in torch_inputs]
            else:
                raise TypeError(f"Got unexpected element type for x[0]={type(x[0])}.")
        return self.check_dim(tensors)

    def check_dim(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Check tensor dimensions.
        Args:
            x (list[torch.Tensor]): List of tensors.
        Returns:
            list[torch.Tensor]: List of tensors with corrected dimensions.
        """
        y = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            elif xi.ndim in (4, 5):
                pass
            else:
                raise ValueError(f"Got unexpected dim for xi={xi.ndim}.")
            y.append(xi)
        return y

    def normalize_split_head(self, x: torch.Tensor, expected_channels: set[int]) -> torch.Tensor:
        """Normalize a split detection head to ``(B, C, H, W)`` layout.

        This accepts the channel-last tensors produced by ONNX export flows as
        well as the channel-first tensors commonly returned by MXQ/NPU inference.

        Args:
            x: Raw split-head tensor.
            expected_channels: Valid channel sizes for the current head group.

        Returns:
            The normalized tensor in ``(B, C, H, W)`` format.

        Raises:
            ValueError: If the tensor shape cannot be interpreted.
        """
        while x.ndim > 4:
            singleton_dims = [idx for idx, size in enumerate(x.shape) if size == 1]
            if not singleton_dims:
                raise ValueError(f"Expected up to 4D split-head tensor, got shape {tuple(x.shape)}.")
            x = x.squeeze(singleton_dims[0])
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D split-head tensor, got shape {tuple(x.shape)}.")

        if x.shape[1] in expected_channels and x.shape[-1] not in expected_channels:
            return x
        if x.shape[-1] in expected_channels and x.shape[1] not in expected_channels:
            return x.permute(0, 3, 1, 2)
        if x.shape[1] in expected_channels and x.shape[-1] in expected_channels:
            return x

        raise ValueError(
            f"Could not infer split-head layout for shape {tuple(x.shape)} with expected channels {expected_channels}."
        )

    @abstractmethod
    def rearrange(self, x: list[torch.Tensor]) -> Any:
        """Rearranges raw model outputs into a task-specific intermediate form.

        Args:
            x: Raw output tensors from the model.

        Returns:
            A task-specific intermediate representation used by ``decode``.
        """

    @abstractmethod
    def decode(self, x: Any) -> Any:
        """Decodes rearranged outputs into a family-specific batched representation.

        Args:
            x: Rearranged output tensors.

        Returns:
            Decoded detections in the canonical representation for that YOLO family.
        """

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters converted outputs into per-image detections before NMS.

        Args:
            x: Converted output tensor.

        Returns:
            Filtered detections for each image in the batch.
        """

    @abstractmethod
    def nms(self, x: Any) -> list[torch.Tensor]:
        """Performs non-maximum suppression on decoded detections.

        Args:
            x: Decoded detections for each image.

        Returns:
            Detections after NMS for each image in the batch.
        """

    def masking(self, x: list[torch.Tensor], proto_outs: torch.Tensor | list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """Apply prototype masks to detection results.

        Args:
            x: Detection results.
            proto_outs: Prototype outputs for masks.

        Returns:
            list: Detection results with masks.
        """
        masks = []
        for pred, proto in zip(x, proto_outs):
            if proto.ndim != 3:
                raise ValueError(f"Expected 3D prototype tensor, got shape {tuple(proto.shape)}.")
            if proto.shape[-1] == self.n_extra:
                proto = proto.permute(2, 0, 1)
            elif proto.shape[0] != self.n_extra:
                raise ValueError(f"Unsupported prototype tensor shape {tuple(proto.shape)}.")
            if len(pred) == 0:
                masks.append(torch.zeros((0, self.imh, self.imw), dtype=torch.float32, device=self.device))
                continue
            masks.append(process_mask_upsample(proto, pred[:, 6:], pred[:, :4], [self.imh, self.imw]))
        return [[xi, mask] for xi, mask in zip(x, masks)]
