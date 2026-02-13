"""
YOLO anchorless postprocessing.
"""

from typing import List

import torch
import torch.nn.functional as F

from .base import YOLOPostBase
from .common import dist2bbox, non_max_suppression, xywh2xyxy


class YOLOAnchorlessPost(YOLOPostBase):
    """Postprocessing for YOLO models without anchors."""

    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
        self.reg_max = post_cfg.get("reg_max", 0)  # DFL channels
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor (144)
        self.dfl_weight = torch.arange(
            self.reg_max, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1, 1)

    def rearrange(
        self,
        x: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Rearranges raw model output tensors into a standardized format.

        Args:
            x (List[torch.Tensor]): List of raw output tensors from the model detection heads.

        Returns:
            List[torch.Tensor]: Rearranged tensors, sorted by size descending.
        """
        y_det = []
        y_cls = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")
            if xi.shape[-1] == self.reg_max * 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(y_det, y_cls)
        ]
        return y

    def decode(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decodes model outputs into box coordinates and class scores.

        Args:
            x (List[torch.Tensor]): Rearranged output tensors from `rearrange`.

        Returns:
            List[torch.Tensor]: Decoded detections for each image in the batch.
        """
        batch_box_cls = torch.cat(x, dim=-1)  # (b, 144, 8400)
        return [self.process_box_cls(box_cls) for box_cls in batch_box_cls]

    def process_box_cls(self, box_cls):
        """
        Process detection results for a single image.
        Args:
            box_cls (torch.Tensor): Raw detections for one image.
        Returns:
            torch.Tensor: Decoded boxes, scores, and extra data.
        """
        if self.n_extra == 0:
            ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.inv_conf_thres
        else:
            ic = (
                torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0)
                > self.inv_conf_thres
            )
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros(
                (0, 4 + self.nc + self.n_extra), dtype=torch.float32
            )  # (0, 84)
        box, scores, extra = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 80, *), (1, 32, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.stride[:, ic]
        )
        return (
            torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0).transpose(0, 1)
        )

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Concatenated output tensor from the model.

        Returns:
            List[torch.Tensor]: Filtered detections for each image in the batch.
        """
        x_list = torch.split(
            x.squeeze(1), 1, dim=0
        )  # [(1, 8400, 84), (1, 8400, 84), ...]

        def process_conversion(x):
            x = x.squeeze(0)  # (8400, 84)
            if self.n_extra == 0:
                ic = torch.amax(x[:, -self.nc :], dim=1) > self.conf_thres
            else:
                ic = (
                    torch.amax(x[:, -self.nc - self.n_extra : -self.n_extra], dim=1)
                    > self.conf_thres
                )
            x = x[ic]
            if x.numel() == 0:
                return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)
            x = xywh2xyxy(x)
            return x

        return [process_conversion(xi) for xi in x_list]

    def nms(
        self,
        x: List[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> List[torch.Tensor]:
        """Performs Non-Maximum Suppression (NMS) on the decoded detections.

        Args:
            x (List[torch.Tensor]): Decoded detections for each image.
            max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            max_nms (int, optional): Maximum number of candidates to consider for NMS.
                Defaults to 30000.
            max_wh (int, optional): Maximum box width/height for offset calculation.
                Defaults to 7680.

        Returns:
            List[torch.Tensor]: Post-NMS detections for each image.
        """
        output = []
        for xi in x:
            if xi.numel() == 0:
                output.append(
                    torch.zeros(
                        (0, 6 + self.n_extra), dtype=torch.float32, device=self.device
                    )
                )
                continue
            box, score, extra = xi[:, :4], xi[:, 4 : 4 + self.nc], xi[:, 4 + self.nc :]
            i, j = (score > self.conf_thres).nonzero(as_tuple=False).T
            xi = torch.cat(
                [box[i], xi[i, j + 4, None], j[:, None].float(), extra[i]], 1
            ).to(self.device)
            if xi.numel() == 0:
                output.append(
                    torch.zeros(
                        (0, 6 + self.n_extra), dtype=torch.float32, device=self.device
                    )
                )
                continue
            xi = xi[torch.argsort(xi[:, 4], descending=True)[:max_nms]]
            # NMS
            c = xi[:, 5:6] * max_wh
            boxes, scores = xi[:, :4] + c, xi[:, 4]
            i = non_max_suppression(boxes, scores, self.iou_thres, max_det)
            output.append(xi[i])
        return output

    def dfl(self, x):
        """
        Applies Distribution Focal Loss projection.
        Args:
            x (Tensor): shape (B, 4 * reg_max, A), where:
                        - B: batch size
                        - reg_max: number of bins per coordinate
                        - A: number of anchor points
        Returns:
            Tensor: shape (B, 4, A), the projected distances.
        """
        if self.reg_max == 0:  # skip dfl for yolov6 s, n models
            return x
        assert x.ndim == 3, "Input must be a 3D tensor (B, 4 * reg_max, A)"
        B, _, A = x.shape
        # Reshape to (B, 4, reg_max, A)
        x = x.view(B, 4, self.reg_max, A).permute(0, 2, 1, 3)
        x = x.softmax(dim=1)
        # dfl_weight: (1, reg_max, 1, 1)
        out = F.conv2d(x, self.dfl_weight, bias=None)  # (B, 1, 4, A)
        # Reshape to (B, 4, A)
        return out.view(B, 4, A)


class YOLOAnchorlessSegPost(YOLOAnchorlessPost):
    """Postprocessing for YOLO segmentation models without anchors."""

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
        if (self.nc + self.n_extra + 4) in x[0].shape[1:] and self.n_extra in x[
            1
        ].shape[1:]:
            return (
                x[0],
                x[1],
            )
        if (self.nc + self.n_extra + 4) in x[1].shape[1:] and self.n_extra in x[
            0
        ].shape[1:]:
            return (
                x[1],
                x[0],
            )
        raise ValueError(f"Wrong shape of input: {x[0].shape}, {x[1].shape}")

    def rearrange(self, x):
        """
        Rearrange model output tensors for segmentation tasks.
        Args:
            x (list[torch.Tensor]): Raw output tensors.
        Returns:
            tuple: (rearranged_detections, prototype_masks)
        """
        y_det = []
        y_cls = []
        y_ext = []
        for xi in x:
            if xi.shape[-1] == self.n_extra:
                y_ext.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 32, 160, 160), (b, 32, 80, 80), ...
            elif xi.shape[-1] == self.reg_max * 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_ext = sorted(y_ext, key=lambda x: x.numel(), reverse=True)
        proto = y_ext.pop(0).permute(0, 2, 3, 1)
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        assert (
            len(y_cls) == len(y_det) == len(y_ext)
        ), "output arguments are not in a proper form"
        y = [
            torch.cat((yi_det, yi_cls, yi_ext), dim=1).flatten(2)
            for (yi_det, yi_cls, yi_ext) in zip(y_det, y_cls, y_ext)
        ]
        return y, proto


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    """Postprocessing for YOLO pose estimation models without anchors."""

    def rearrange(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Rearranges model output tensors for pose estimation tasks.

        Args:
            x (List[torch.Tensor]): Raw output tensors.

        Returns:
            List[torch.Tensor]: Rearranged tensors.
        """
        y_det = []
        y_cls = []
        y_kpt = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")
            if xi.shape[-1] == self.reg_max * 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 1, 80, 80), (b, 1, 40, 40), ...
            elif xi.shape[-1] == self.n_extra:
                y_kpt.append(
                    xi.permute(0, 3, 1, 2).flatten(2)
                )  # (b, 51, 80, 80), (b, 1, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y_kpt = sorted(
            y_kpt, key=lambda x: x.numel(), reverse=True
        )  # (b, 51, 6400), (b, 51, 1600), (b, 51, 400)
        y_tmp = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(
                y_det, y_cls
            )  # (b, 65, 6400), (b, 65, 1600), (b, 65, 400)
        ]
        y = [
            torch.cat((yi_tmp, yi_kpt), dim=1) for (yi_tmp, yi_kpt) in zip(y_tmp, y_kpt)
        ]  # (b, 116, 6400), (b, 116, 1600), (b, 116, 400)
        return y

    def process_box_cls(self, box_cls):
        """
        Process pose estimation results for a single image.
        Args:
            box_cls (torch.Tensor): Raw detections for one image.
        Returns:
            torch.Tensor: Decoded boxes, scores, and keypoints.
        """
        ic = (
            torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0)
            > self.inv_conf_thres
        )
        box_cls = box_cls[:, ic]  # (116, *)
        if box_cls.numel() == 0:
            return torch.zeros(
                (0, 4 + self.nc + self.n_extra), dtype=torch.float32
            )  # (0, 56)
        box, scores, keypoints = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 1, *), (1, 51, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.stride[:, ic]
        )
        keypoints = keypoints.view(1, 17, 3, -1)
        key_coord, key_conf = torch.split(
            keypoints, [2, 1], dim=2
        )  # (1, 17, 2, 8400), (1, 17, 1, 8400)
        key_coord = (key_coord * 2 + self.anchors[:, ic] - 0.5) * self.stride[
            :, ic
        ]  # (1, 17, 2, *)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(
            1, self.n_extra, -1
        )  # (1, 51, *)
        return (
            torch.cat([dbox, scores.sigmoid(), keypoints], dim=1)
            .squeeze(0)
            .transpose(0, 1)
        )  # (*, 56)
