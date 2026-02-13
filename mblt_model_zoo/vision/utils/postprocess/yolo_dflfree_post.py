from typing import List

import torch

from .base import YOLOPostBase
from .common import dist2bbox, dual_topk


class YOLODFLFreePost(YOLOPostBase):
    """Postprocessing for YOLO DFL-free models."""

    def __call__(self, x, conf_thres: float, iou_thres: float) -> List[torch.Tensor]:
        """Executes YOLO postprocessing for DFL-free models.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model outputs.
            conf_thres (float): Confidence threshold.
            iou_thres (float): IoU threshold.

        Returns:
            List[torch.Tensor]: List of detections per image.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        if len(x) == 2:
            x = self.conversion(x)
            x = self.filter_conversion(x)
        else:
            x = self.rearrange(x)
            x = self.decode(x)
        x = self.nms(x)
        return x

    def conversion(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Converts raw model output tensors into a single concatenated tensor.

        Args:
            x (List[torch.Tensor]): List of raw output tensors.

        Returns:
            torch.Tensor: Concatenated tensor of shape (batch, num_anchors, 4 + nc + n_extra).
        """
        assert (
            len(x) == 2
        ), f"Assume return is a list of two outputs, but got {len(x)} outputs"
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=False)
        return torch.cat(x, dim=-1).squeeze(1)  # [b, 8400, 84]

    def rearrange(self, x):
        y_det = []
        y_cls = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")
            if xi.shape[-1] == 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 4, 80, 80), (b, 4, 40, 40), ...
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
        batch_box_cls = torch.cat(x, dim=-1)  # (b, 84, 8400)
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
        box_cls = box_cls[:, ic]  # (84, *)
        if box_cls.numel() == 0:
            return torch.zeros(
                (0, 4 + self.nc + self.n_extra), dtype=torch.float32
            )  # (0, 84)
        box, scores, extra = torch.split(
            box_cls[None], [4, self.nc, self.n_extra], dim=1
        )  # (*, 4), (*, 80), (*, 32)
        dbox = (
            dist2bbox(
                box,
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.stride[:, ic]
        )
        pre_topk = (
            torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0).transpose(0, 1)
        )  # (*, 84)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres)

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Output tensor from the model.

        Returns:
            List[torch.Tensor]: Filtered detections for each image in the batch.
        """
        x_list = torch.split(x, 1, dim=0)  # [(1, 8400, 84), (1, 8400, 84), ...]

        def process_conversion(x):
            x = x.squeeze(0)
            if self.n_extra == 0:
                ic = torch.amax(x[..., -self.nc :], dim=-1) > self.conf_thres
            else:
                ic = (
                    torch.amax(x[..., -self.nc - self.n_extra : -self.n_extra], dim=-1)
                    > self.conf_thres
                )
            pre_topk = x[ic]  # (*, 84)
            return dual_topk(
                pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres
            )

        return [process_conversion(xi) for xi in x_list]

    def nms(
        self,
        x: List[torch.Tensor],
        _max_det: int = 300,
        _max_nms: int = 30000,
        _max_wh: int = 7680,
    ) -> List[torch.Tensor]:
        """Performs Non-Maximum Suppression (no-op for NMS-free models).

        Args:
            x (List[torch.Tensor]): Decoded detections.
            _max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            _max_nms (int, optional): Maximum candidates for NMS. Defaults to 30000.
            _max_wh (int, optional): Maximum box width/height. Defaults to 7680.

        Returns:
            List[torch.Tensor]: The input detections unchanged.
        """
        return x


class YOLODFLFreeSegPost(YOLODFLFreePost):
    """Postprocessing for YOLO NMS-free segmentation models."""

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
        if len(x) == 4:
            x, proto_outs = self.conversion(x)
            x = self.filter_conversion(x)
        else:
            x, proto_outs = self.rearrange(x)
            x = self.decode(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def conversion(self, x: List[torch.Tensor]):
        """Convert input tensors.
        Args:
            x (List[torch.Tensor]): Input tensors.
        Returns:
            tuple:
                - outputs (torch.Tensor): Processed outputs.
                - proto (torch.Tensor): Prototype masks.
        """
        assert (
            len(x) == 4
        ), f"Assume return is a list of four outputs, but got {len(x)} outputs"
        x = sorted(x, key=lambda x: x.size(), reverse=False)
        outputs = []
        protos = []
        for xi in x:
            if xi.shape[-1] == self.n_extra:
                protos.append(xi)
            else:
                outputs.append(xi)
        proto = protos.pop(-1)
        outputs = torch.cat(outputs + protos, dim=-1).squeeze(1)
        return outputs, proto

    def rearrange(self, x):
        y_det = []
        y_cls = []
        y_ext = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")
            if xi.shape[-1] == self.n_extra:
                y_ext.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 32, 160, 160), (b, 32, 80, 80), ...
            elif xi.shape[-1] == 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 4, 80, 80), (b, 4 ,40, 40), ...
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


class YOLODFLFreePosePost(YOLODFLFreePost):
    """Postprocessing for YOLO NMS-free pose estimation models."""

    def __call__(self, x, conf_thres, iou_thres):
        """Execute YOLO postprocessing.
        Args:
            x: Input tensor or list of tensors.
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.
        Returns:
            list: Postprocessed results.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        if len(x) == 3:
            x = self.conversion(x)
            x = self.filter_conversion(x)
        else:
            x = self.rearrange(x)
            x = self.decode(x)
        x = self.nms(x)
        return x

    def conversion(self, x: List[torch.Tensor]):
        """Convert input tensors.
        Args:
            x (List[torch.Tensor]): Input tensors.
        Returns:
            torch.Tensor: Converted tensor.
        """
        assert (
            len(x) == 3
        ), f"Assume return is a list of three outputs, but got {len(x)} outputs"
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=True)
        kpt = x.pop(0)
        kpt = kpt.permute(0, -2, -3, -1).flatten(-2)
        return torch.cat(
            [torch.cat(x, dim=-1).squeeze(1), kpt], dim=-1
        )  # [b, 8400, 56]

    def rearrange(self, x):
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
            if xi.shape[-1] == 4:
                y_det.append(
                    xi.permute(0, 3, 1, 2)
                )  # (b, 4, 80, 80), (b, 4 ,40, 40), ...
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
            box_cls[None], [4, self.nc, self.n_extra], dim=1
        )  # (1, 4, *), (1, 1, *), (1, 51, *)
        dbox = (
            dist2bbox(
                box,
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
        key_coord = (key_coord + self.anchors[:, ic]) * self.stride[
            :, ic
        ]  # (1, 17, 2, *)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(
            1, self.n_extra, -1
        )  # (1, 51, *)
        pre_topk = (
            torch.cat([dbox, scores.sigmoid(), keypoints], dim=1)
            .squeeze(0)
            .transpose(0, 1)
        )  # (*, 56)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres)
