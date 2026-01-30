from typing import List

import torch

from .base import YOLOPostBase
from .common import non_max_suppression, xywh2xyxy


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convert the output of ONNX model to the input of NMS.

        Args:
            x: NPU outputs
        Returns:
            Decoded outputs
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

    def nms(self, x, max_det=300, max_nms=30000, max_wh=7680):
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


class YOLOAnchorlessSegPost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(
            pre_cfg,
            post_cfg,
        )

    def __call__(self, x, conf_thres=None, iou_thres=None):
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        x, proto_outs = self.conversion(x)
        x = self.filter_conversion(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def conversion(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert the output of ONNX model to the input of NMS.
        Args:
            npu_outputs: list of np arrays, NPU outputs
        Returns:
            outputs decoded outputs(boxes, conf, scores, keypts/lmarks/masks)
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
