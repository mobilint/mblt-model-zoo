from typing import List

import torch

from .base import YOLOPostBase
from .common import non_max_suppression, xywh2xyxy


class YOLOAnchorPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
        self.no = self.nc + 5 + self.n_extra

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convert the output of ONNX model to the input of NMS.

        Args:
            x: NPU outputs
        Returns:
            Decoded outputs
        """
        x_list = torch.split(
            x.squeeze(1), 1, dim=0
        )  # [(1, 25200, 85), (1, 25200, 85), ...]

        def process_conversion(x):
            x = x.squeeze(0)  # (25200, 85)
            ic = x[:, 4] > self.conf_thres  # candidates
            x = x[ic]  # (n, 85)

            if len(x) == 0:
                return torch.zeros((0, 5 + self.nc + self.n_extra), dtype=torch.float32)

            return x

        return [process_conversion(xi) for xi in x_list]

    def nms(self, x, max_det=300, max_nms=30000, max_wh=7680):
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
            xi = torch.cat(
                [box[i], xi[i, j + 5, None], j[:, None].float(), mask[i]], 1
            ).to(self.device)

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


class YOLOAnchorSegPost(YOLOAnchorPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

    def __call__(self, x, conf_thres=None, iou_thres=None):
        self.set_threshold(conf_thres, iou_thres)
        x = self.check_input(x)
        x, proto_outs = self.conversion(x)
        x = self.filter_conversion(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def conversion(self, x: List[torch.Tensor]):
        """
        Convert NPU outputs from ONNX inference to evaluable format.
        Args:
            npu_outs: list of np arrays, NPU(shouldn't contain decode) outputs
        Returns:
            proto_out: proto output(np.array)
            npu_outs: npu output(concatenated np.array)
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
