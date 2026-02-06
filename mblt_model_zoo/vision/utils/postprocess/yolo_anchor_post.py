"""
YOLO anchor-based postprocessing.
"""

from typing import List

import torch

from .base import YOLOPostBase
from .common import non_max_suppression, xywh2xyxy


class YOLOAnchorPost(YOLOPostBase):
    """Postprocessing for YOLO models with anchors."""

    def __init__(self, pre_cfg: dict, post_cfg: dict):
        """Initialize YOLOAnchorPost.
        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        super().__init__(pre_cfg, post_cfg)
        self.no = self.nc + 5 + self.n_extra
        self.make_anchor_grid()

    def rearrange(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Rearranges raw model output tensors into a standardized channel-first format.

        Args:
            x (List[torch.Tensor]): Raw output tensors from the model detection heads.

        Returns:
            List[torch.Tensor]: Rearranged tensors in (batch, channels, height, width) format,
                sorted by image size descending.
        """
        assert (
            len(x) == self.nl
        ), f"Got unsupported number of detection heads: {len(x)}."
        y = []
        for i in range(self.nl):
            tmp = x[i]
            if tmp.shape[3] == self.no * self.na:
                y.append(
                    tmp.permute(0, 3, 1, 2)
                )  # (b, 80, 80, 255) -> (b, 255, 80, 80)
            else:
                raise NotImplementedError(
                    f"Got unsupported shape for input: {tmp.shape}."
                )
        # sort by image size descending
        y = sorted(y, key=lambda x: x.numel(), reverse=True)
        return y

    def decode(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decodes model outputs into box coordinates and class scores.

        Applies sigmoid to predictions and transforms boxes from anchor-relative
        to image-relative coordinates.

        Args:
            x (List[torch.Tensor]): Rearranged output tensors from `rearrange`.

        Returns:
            List[torch.Tensor]: Decoded detections for each image in the batch.
        """
        batch_box_cls = torch.cat(
            [
                xi.reshape(xi.shape[0], self.na, self.no, xi.shape[-2], xi.shape[-1])
                .permute(0, 1, 3, 4, 2)
                .reshape(xi.shape[0], -1, self.no)
                for xi in x
            ],
            dim=1,
        )  # (bs, 25200, 85)
        return [self.process_box_cls(box_cls) for box_cls in batch_box_cls]

    def process_box_cls(self, x):
        """
        Process a single image's detection tensor.
        Args:
            x (torch.Tensor): Raw detections for one image.
        Returns:
            torch.Tensor: Decoded boxes, confidence, and scores.
        """
        ic = x[:, 4] > self.inv_conf_thres  # candidates
        box_cls = x[ic]  # (n, 85)
        if len(box_cls) == 0:
            return torch.zeros((0, 5 + self.nc + self.n_extra), dtype=torch.float32)
        grid = self.grid[ic, :]  # (n, 2)
        anchor_grid = self.anchor_grid[ic, :]  # (n, 2)
        stride = self.stride[ic, :]  # (n, 2)
        xy, wh, conf, scores, extra = self.chop(
            box_cls
        )  # (n, 2), (n, 2), (n, 1), (n, 80), (n, 0 or 32)
        xy = (xy.sigmoid() * 2 - 0.5 + grid) * stride
        wh = (wh.sigmoid() * 2) ** 2 * anchor_grid
        return torch.cat((xy, wh, conf.sigmoid(), scores.sigmoid(), extra), dim=1)

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Concatenated output tensor from the model.

        Returns:
            List[torch.Tensor]: Filtered detections for each image in the batch.
        """
        x_list = torch.split(
            x.squeeze(1), 1, dim=0
        )  # [(1, 25200, 85), (1, 25200, 85), ...]

        def process_conversion(x):
            x = x.squeeze(0)  # (25200, 85)
            ic = x[:, 4] > self.conf_thres  # candidates
            x = x[ic]  # (n, 85)
            if len(x) == 0:
                return torch.zeros((0, self.no), dtype=torch.float32)
            return x

        return [process_conversion(xi) for xi in x_list]

    def nms(self, x, max_det=300, max_nms=30000, max_wh=7680):
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
                output.append(
                    torch.zeros(
                        (0, 6 + self.n_extra), dtype=torch.float32, device=self.device
                    )
                )
                continue
            xi[..., 5:] *= xi[..., 4:5]  # conf = obj_conf * cls_conf
            box = xywh2xyxy(xi[..., :4])
            mask = xi[..., mi:]
            i, j = (xi[..., 5:mi] > self.conf_thres).nonzero(as_tuple=False).T
            xi = torch.cat([box[i], xi[i, j + 5, None], j[:, None].float(), mask[i]], 1)
            if xi.numel() == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue
            xi = xi[xi[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # NMS
            c = xi[:, 5:6] * max_wh
            boxes, scores = xi[:, :4] + c, xi[:, 4]
            i = non_max_suppression(boxes, scores, self.iou_thres, max_det)
            output.append(xi[i])
        return output

    def make_anchor_grid(self):
        """
        Pre-calculate the anchor grid for decoding.
        """
        self.grid, self.anchor_grid, self.stride = [], [], []
        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]
        out_sizes = [
            [self.imh // strd, self.imw // strd] for strd in strides
        ]  # (80, 80), (40, 40), (20, 20)
        for anchr, (ny, nx), strd in zip(self.anchors, out_sizes, strides):
            yv, xv = torch.meshgrid(
                torch.arange(ny, dtype=torch.float32, device=self.device),
                torch.arange(nx, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).expand(self.na, ny, nx, 2)
            self.grid.append(grid)
            anchr = torch.broadcast_to(
                torch.tensor(anchr).reshape(self.na, 1, 1, 2),
                (self.na, ny, nx, 2),
            )
            self.anchor_grid.append(anchr)
            self.stride.append(strd * torch.ones(self.na, ny, nx, 2))
        self.grid = torch.cat([grd.reshape(-1, 2) for grd in self.grid], dim=0)
        self.anchor_grid = torch.cat(
            [anc.reshape(-1, 2) for anc in self.anchor_grid], dim=0
        )
        self.stride = torch.cat([strd.reshape(-1, 2) for strd in self.stride], dim=0)

    def chop(self, npu_out: torch.Tensor, idx: int = 0) -> tuple:
        """Splits the detection tensor into individual components (xy, wh, conf, scores, extra).

        Args:
            npu_out (torch.Tensor): Raw detection tensor from one detection head.
            idx (int, optional): Detection head index. Defaults to 0.

        Returns:
            tuple: (xy, wh, conf, scores, extra).
        """
        xy, wh, conf, scores, extra = torch.split(
            npu_out, [2, 2, 1, self.nc, self.n_extra], dim=-1
        )
        return xy, wh, conf, scores, extra


class YOLOAnchorSegPost(YOLOAnchorPost):
    """Postprocessing for YOLO segmentation models with anchors."""

    def __call__(self, x, conf_thres, iou_thres):
        """Execute YOLO segmentation postprocessing.
        Args:
            x: Input tensor or list of tensors.
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.
        Returns:
            list: Postprocessed results with masks.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        if len(x) == 2:
            x, proto_outs = self.conversion(x)
            x = self.filter_conversion(x)
        else:
            x, proto_outs = self.rearrange(x)
            x = self.decode(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def conversion(self, x: List[torch.Tensor]) -> tuple:
        """Converts raw model output tensors into detections and prototypes.

        Args:
            x (List[torch.Tensor]): List of raw output tensors.

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

    def rearrange(self, x: List[torch.Tensor]) -> tuple:
        """Rearranges model output tensors for segmentation tasks.

        Args:
            x (List[torch.Tensor]): Raw output tensors from detection and prototype heads.

        Returns:
            tuple: (rearranged_detections, prototype_masks)
        """
        proto = None
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
        return y, proto

    def chop(self, npu_out: torch.Tensor, idx: int = 0) -> tuple:
        """Splits the detection tensor for segmentation tasks.

        Args:
            npu_out (torch.Tensor): Raw detection tensor.
            idx (int, optional): Detection head index. Defaults to 0.

        Returns:
            tuple: (xy, wh, conf, scores, masks).
        """
        xy, wh, conf, scores, masks = torch.split(
            npu_out, [2, 2, 1, self.nc, self.n_extra], dim=-1
        )
        masks = masks * conf.sigmoid()
        return xy, wh, conf, scores, masks
