"""
YOLO NMS-free postprocessing.
"""

from typing import List

import torch

from .common import dist2bbox, dual_topk
from .yolo_anchorless_post import YOLOAnchorlessPost


class YOLONMSFreePost(YOLOAnchorlessPost):
    """Postprocessing for YOLO NMS-free models."""

    def __call__(self, x, conf_thres: float, iou_thres: float) -> List[torch.Tensor]:
        """Executes YOLO postprocessing for NMS-free models.

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

    def conversion(self, x: List[torch.Tensor]):
        """Convert input tensors.
        Args:
            x (List[torch.Tensor]): Input tensors.
        Returns:
            torch.Tensor: Converted tensor.
        """
        assert len(x) == 2, f"Expected 2 output tensors, but got {len(x)}"
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=False)
        return torch.cat(x, dim=-1).squeeze(1)  # [b, 8400, 84]

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Filters out low-confidence detections from a single output tensor.

        Args:
            x (torch.Tensor): Model output tensor.

        Returns:
            List[torch.Tensor]: Decoded and filtered outputs for each image.
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

    def process_box_cls(self, box_cls):
        """
        Process detection results for a single image.
        Args:
            box_cls (torch.Tensor): Raw detections for one image.
        Returns:
            torch.Tensor: Decoded and top-k filtered detections.
        """
        ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros((0, 6), dtype=torch.float32)  # (0, 6)
        box, scores = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc], dim=1
        )  # (1, 64, *), (1, 80, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.stride[:, ic]
        )
        pre_topk = (
            torch.cat([dbox, scores.sigmoid()], dim=1).squeeze(0).transpose(0, 1)
        )  # (*, 84)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres)

    def nms(
        self,
        x: List[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> List[torch.Tensor]:
        """Perform Non-Maximum Suppression (no-op for NMS-free models).

        Args:
            x (List[torch.Tensor]): Decoded detections.
            max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            max_nms (int, optional): Maximum candidates for NMS. Defaults to 30000.
            max_wh (int, optional): Maximum box width/height. Defaults to 7680.

        Returns:
            List[torch.Tensor]: The input detections unchanged.
        """
        return x
