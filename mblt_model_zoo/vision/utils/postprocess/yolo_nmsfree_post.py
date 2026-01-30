"""
YOLO NMS-free postprocessing.
"""

from typing import List

import torch

from .yolo_anchorless_post import YOLOAnchorlessPost


class YOLONMSFreePost(YOLOAnchorlessPost):
    """Postprocessing for YOLO NMS-free models."""

    def conversion(self, x: List[torch.Tensor]):
        """Convert input tensors.

        Args:
            x (List[torch.Tensor]): Input tensors.

        Returns:
            torch.Tensor: Converted tensor.
        """
        assert (
            len(x) == 2
        ), f"Assume return is a list of two outputs, but got {len(x)} outputs"
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=False)
        return torch.cat(x, dim=-1).squeeze(1)  # [b, 8400, 84]

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convert the output of NPU to the input of NMS.

        Args:
            x (torch.Tensor): NPU outputs

        Returns:
            List[torch.Tensor]: Decoded outputs
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
            max_det = min(pre_topk.shape[0], 300)

            # first topk
            box, scores, extra = pre_topk.split([4, self.nc, self.n_extra], dim=-1)
            max_scores = scores.amax(dim=-1)
            max_scores, index = torch.topk(
                max_scores, max_det, dim=-1
            )  # max deteciton is 300
            index = index.unsqueeze(-1)
            box = torch.gather(box, dim=0, index=index.repeat(1, 4))
            scores = torch.gather(scores, dim=0, index=index.repeat(1, self.nc))
            extra = torch.gather(extra, dim=0, index=index.repeat(1, self.n_extra))

            # second topk
            scores, index = torch.topk(scores.flatten(), max_det)
            index = index.unsqueeze(-1)
            scores = scores.unsqueeze(-1)
            labels = index % self.nc
            index = index // self.nc
            box = box.gather(dim=0, index=index.repeat(1, 4))
            extra = extra.gather(dim=0, index=index.repeat(1, self.n_extra))

            box_cls = torch.cat([box, scores, labels, extra], dim=1)  # (300, 6)

            box_cls = box_cls[box_cls[:, 4] > self.conf_thres]  # final filtering
            if box_cls.numel() == 0:
                return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32)

            return box_cls

        return [process_conversion(xi) for xi in x_list]

    def nms(
        self, x, max_det=300, max_nms=30000, max_wh=7680
    ):  # Do nothing on NMS Free model
        """Perform Non-Maximum Suppression (No-op).

        Args:
            x: Input tensor.
            max_det (int, optional): Maximum number of detections. Defaults to 300.
            max_nms (int, optional): Maximum number of boxes for NMS. Defaults to 30000.
            max_wh (int, optional): Maximum width/height for NMS. Defaults to 7680.

        Returns:
            torch.Tensor: Input tensor.
        """
        return x


class YOLONMSFreeSegPost(YOLONMSFreePost):
    """Postprocessing for YOLO NMS-free segmentation models."""

    def __call__(self, x, conf_thres=None, iou_thres=None):
        """Execute YOLO NMS-free segmentation postprocessing.

        Args:
            x: Input tensor or list of tensors.
            conf_thres (float, optional): Confidence threshold.
            iou_thres (float, optional): IoU threshold.

        Returns:
            list: Postprocessed results with masks.
        """
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        x, proto_outs = self.conversion(x)
        x = self.filter_conversion(x)
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


class YOLONMSFreePosePost(YOLONMSFreePost):
    """Postprocessing for YOLO NMS-free pose estimation models."""

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
        kpt = kpt.permute(0, -1, -3, -2).flatten(-2)
        return torch.cat(
            [torch.cat(x, dim=-1).squeeze(1), kpt], dim=-1
        )  # [b, 8400, 56]
