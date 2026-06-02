"""
YOLO anchor-based postprocessing.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from .base import YOLOPostBase
from .common import YOLOSegPostMixin, non_max_suppression


class YOLOAnchorPost(YOLOPostBase):
    """Postprocessing for YOLO models with anchors."""

    def __init__(self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any], **kwargs: Any) -> None:
        """Initialize YOLOAnchorPost.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)
        self.no = self.nc + 5 + self.n_extra
        self.grid: torch.Tensor
        self.anchor_grid: torch.Tensor
        self.make_anchor_grid()

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rearranges raw model output tensors into a concatenated decode input.

        Args:
            x (list[torch.Tensor]): Raw output tensors from the model detection heads.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Concatenated tensor in
                ``(batch, anchors, no)`` format, optionally paired with prototype masks in
                segmentation subclasses.
        """
        assert len(x) == self.nl, f"Got unsupported number of detection heads: {len(x)}."
        y = []
        for i in range(self.nl):
            tmp = x[i]
            if tmp.shape[3] == self.no * self.na:
                y.append(tmp.permute(0, 3, 1, 2))  # (b, 80, 80, 255) -> (b, 255, 80, 80)
            else:
                raise NotImplementedError(f"Got unsupported shape for input: {tmp.shape}.")
        # sort by image size descending
        y = sorted(y, key=lambda x: x.numel(), reverse=True)
        return torch.cat(
            [
                xi.reshape(xi.shape[0], self.na, self.no, xi.shape[-2], xi.shape[-1])
                .permute(0, 1, 3, 4, 2)
                .reshape(xi.shape[0], -1, self.no)
                for xi in y
            ],
            dim=1,
        )

    def decode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Decodes model outputs into box coordinates and class scores.

        Applies sigmoid to predictions and transforms boxes from anchor-relative
        to image-relative coordinates.

        Args:
            x (torch.Tensor): Concatenated output tensor from `rearrange`.

        Returns:
            list[torch.Tensor]: Decoded detections for each image in the batch.
        """
        return [self.process_box_cls(box_cls) for box_cls in x]

    def process_box_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a single image's detection tensor.

        Args:
            x: Raw detections for one image.

        Returns:
            Decoded boxes, confidence, and scores.
        """
        ic = x[:, 4] > self.inv_conf_thres  # candidates
        box_cls = x[ic]  # (n, 85)
        if box_cls.numel() == 0:
            return torch.zeros((0, 5 + self.nc + self.n_extra), dtype=torch.float32, device=x.device)

        grid = self.grid[ic, :]  # (n, 2)
        anchor_grid = self.anchor_grid[ic, :]  # (n, 2)
        stride = self.stride_as_tensor()[ic, :]  # (n, 2)

        # Advanced indexing above materializes ``box_cls``, so in-place decode avoids a second output allocation.
        box_cls[:, :2] = box_cls[:, :2].sigmoid_().mul_(2.0).add_(grid).add_(-0.5).mul_(stride)
        box_cls[:, 2:4] = box_cls[:, 2:4].sigmoid_().mul_(2.0).pow_(2.0).mul_(anchor_grid)
        conf = box_cls[:, 4:5].sigmoid_()
        box_cls[:, 5 : 5 + self.nc].sigmoid_()
        if self.task == "instance_segmentation" and self.n_extra > 0:
            box_cls[:, 5 + self.nc :] *= conf
        return box_cls

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Concatenated output tensor from the model.

        Returns:
            list[torch.Tensor]: Filtered detections for each image in the batch.
        """
        x_list = torch.split(x.squeeze(1), 1, dim=0)  # [(1, 25200, 85), (1, 25200, 85), ...]

        def process_conversion(x: torch.Tensor) -> torch.Tensor:
            x = x.squeeze(0)  # (25200, 85)
            ic = x[:, 4] > self.conf_thres  # candidates
            x = x[ic]  # (n, 85)
            if len(x) == 0:
                return torch.zeros((0, self.no), dtype=torch.float32)
            return x

        return [process_conversion(xi) for xi in x_list]

    def nms(
        self, x: list[torch.Tensor], max_det: int = 300, max_nms: int = 30000, max_wh: int = 7680
    ) -> list[torch.Tensor]:
        """
        Perform Non-Maximum Suppression (NMS) on the decoded detections.
        Args:
            x (list[torch.Tensor]): Decoded detections for each image.
            max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            max_nms (int, optional): Maximum number of candidates to consider for NMS.
                Defaults to 30000.
            max_wh (int, optional): Maximum box width/height for offset calculation.
                Defaults to 7680.
        Returns:
            list[torch.Tensor]: Post-NMS detections for each image.
        """
        mi = 5 + self.nc  # mask index
        output = []
        for xi in x:
            if xi.numel() == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32, device=self.device))
                continue
            xi[..., 5:] *= xi[..., 4:5]  # conf = obj_conf * cls_conf
            match_index = (xi[..., 5:mi] > self.conf_thres).nonzero(as_tuple=False)
            if match_index.numel() == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue
            i = match_index[:, 0]
            j = match_index[:, 1]
            rows = xi[i]
            boxes_xywh = rows[:, :4]
            xi = torch.empty((rows.shape[0], 6 + self.n_extra), dtype=rows.dtype, device=rows.device)
            xi[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            xi[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            xi[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            xi[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
            xi[:, 4] = rows[torch.arange(rows.shape[0], device=rows.device), j + 5]
            xi[:, 5] = j.to(rows.dtype)
            if self.n_extra > 0:
                xi[:, 6:] = rows[:, mi:]
            xi = xi[xi[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # NMS
            c = xi[:, 5:6] * max_wh
            boxes, scores = xi[:, :4] + c, xi[:, 4]
            i_idx = non_max_suppression(boxes, scores, self.iou_thres, max_det)
            output.append(xi[i_idx])
        return output

    def make_anchor_grid(self) -> None:
        """
        Pre-calculate the anchor grid for decoding.
        """
        grid_parts: list[torch.Tensor] = []
        anchor_grid_parts: list[torch.Tensor] = []
        stride_parts: list[torch.Tensor] = []
        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]
        out_sizes = [[self.imh // strd, self.imw // strd] for strd in strides]  # (80, 80), (40, 40), (20, 20)
        for anchr, (ny, nx), strd in zip(self.anchors_as_list(), out_sizes, strides):
            yv, xv = torch.meshgrid(
                torch.arange(ny, dtype=torch.float32, device=self.device),
                torch.arange(nx, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).expand(self.na, ny, nx, 2)
            grid_parts.append(grid)
            anchr_tensor = torch.broadcast_to(
                torch.tensor(anchr).reshape(self.na, 1, 1, 2),
                (self.na, ny, nx, 2),
            )
            anchor_grid_parts.append(anchr_tensor)
            stride_parts.append(strd * torch.ones(self.na, ny, nx, 2))
        self.grid = torch.cat([grd.reshape(-1, 2) for grd in grid_parts], dim=0)
        self.anchor_grid = torch.cat([anc.reshape(-1, 2) for anc in anchor_grid_parts], dim=0)
        self.stride = torch.cat([strd.reshape(-1, 2) for strd in stride_parts], dim=0)

    def chop(self, npu_out: torch.Tensor, idx: int = 0) -> tuple[torch.Tensor, ...]:
        """Splits the detection tensor into individual components (xy, wh, conf, scores, extra).

        Args:
            npu_out (torch.Tensor): Raw detection tensor from one detection head.
            idx (int, optional): Detection head index. Defaults to 0.

        Returns:
            tuple: (xy, wh, conf, scores, extra).
        """
        xy, wh, conf, scores, extra = torch.split(npu_out, [2, 2, 1, self.nc, self.n_extra], dim=-1)
        return xy, wh, conf, scores, extra


class YOLOAnchorSegPost(YOLOSegPostMixin, YOLOAnchorPost):
    """Postprocessing for YOLO segmentation models with anchors."""

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Preprocesses intermediate inputs into (boxes, proto) format.

        Args:
            x (list[torch.Tensor]): Raw model output tensors.

        Returns:
            tuple: (decoded_detections, prototype_masks).
        """
        if len(x) == 2:
            converted, proto_outs = cast(tuple[torch.Tensor, torch.Tensor], self.conversion(x))
            return self.filter_conversion(converted), proto_outs
        rearranged, proto_outs = self.rearrange(x)
        return self.decode(rearranged), proto_outs

    def conversion(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts raw model output tensors into detections and prototypes.

        Args:
            x (list[torch.Tensor]): List of raw output tensors.

        Returns:
            tuple: (detections, prototypes)
        """
        if self.no in x[0].shape[1:] and self.n_extra in x[1].shape[1:]:
            return (
                x[0],
                x[1],
            )
        if self.n_extra in x[0].shape[1:] and self.no in x[1].shape[1:]:
            return (
                x[1],
                x[0],
            )
        raise NotImplementedError(f"Input shape {x[0].shape} not supported.")

    def rearrange(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rearranges model output tensors for segmentation tasks.

        Args:
            x (list[torch.Tensor]): Raw output tensors from detection and prototype heads.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Concatenated detections and prototype masks.
        """
        proto: torch.Tensor | None = None
        for i, xi in enumerate(x):
            if self.n_extra == xi.shape[-1]:
                proto = x.pop(i)
                break
        if proto is None:
            raise ValueError("Proto output is missing.")
        y = []
        for xi in x:
            if xi.shape[-1] == self.no * self.nl:
                y.append(xi.permute(0, 3, 1, 2))
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort by image size descending
        y = sorted(y, key=lambda x: x.numel(), reverse=True)
        return (
            torch.cat(
                [
                    xi.reshape(xi.shape[0], self.na, self.no, xi.shape[-2], xi.shape[-1])
                    .permute(0, 1, 3, 4, 2)
                    .reshape(xi.shape[0], -1, self.no)
                    for xi in y
                ],
                dim=1,
            ),
            proto,
        )

    def chop(self, npu_out: torch.Tensor, idx: int = 0) -> tuple[torch.Tensor, ...]:
        """Splits the detection tensor for segmentation tasks.

        Args:
            npu_out (torch.Tensor): Raw detection tensor.
            idx (int, optional): Detection head index. Defaults to 0.

        Returns:
            tuple: (xy, wh, conf, scores, masks).
        """
        xy, wh, conf, scores, masks = torch.split(npu_out, [2, 2, 1, self.nc, self.n_extra], dim=-1)
        masks = masks * conf.sigmoid()
        return xy, wh, conf, scores, masks
