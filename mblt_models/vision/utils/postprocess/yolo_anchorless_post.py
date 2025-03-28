import torch
import torchvision
from .base import YOLOPostBase
from .common import *


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
        self.stride = [2 ** (3 + i) for i in range(self.nl)]
        self.make_anchors()
        self.reg_max = 16  # DFL channels
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor (144)

    def make_anchors(self, offset=0.5):
        anchor_points, stride_tensor = [], []
        for strd in self.stride:
            ny, nx = self.imh // strd, self.imw // strd
            sy = torch.arange(ny, dtype=torch.float32) + offset
            sx = torch.arange(nx, dtype=torch.float32) + offset
            yv, xv = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
            stride_tensor.append(torch.full((ny * nx, 1), strd, dtype=torch.float32))

        self.anchors = torch.cat(anchor_points, dim=0).permute(1, 0)
        self.stride = torch.cat(stride_tensor, dim=0).permute(1, 0)

    def rearrange(self, x):
        y = []
        for xi in x:
            if xi.shape[1] == 1:  # (b, 1, 4, 8400)
                xi = xi.squeeze(1)
                xi = xi * self.stride
            elif xi.shape[2] == 1:  # (b, 80, 1, 8400)
                xi = xi.squeeze(2)
            else:
                raise ValueError(f"Got unexpected shape for x={x.shape}.")
            y.append(xi)

        y = sorted(y, key=lambda x: x.numel())
        return torch.cat(y, dim=1)  # (b, 84, 8400)

    def decode(self, x):
        x = x.permute(0, 2, 1)
        x = xywh2xyxy(x)

        y = []
        for xi in x:
            if self.n_extra == 0:
                ic = torch.amax(xi[..., -self.nc :], dim=-1) > self.conf_thres

            else:
                ic = (
                    torch.amax(xi[..., -self.nc - self.n_extra : -self.n_extra], dim=-1)
                    > self.conf_thres
                )

            xi = xi[ic]
            if len(xi) == 0:
                y.append(
                    torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)
                )
                continue
            else:
                y.append(xi)

        return y

    def nms(self, x, max_det=300, max_nms=30000, max_wh=7680):
        output = []

        for xi in x:
            if len(xi) == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue

            box, score, extra = xi[:, :4], xi[:, 4 : 4 + self.nc], xi[:, 4 + self.nc :]
            i, j = (score > self.conf_thres).nonzero(as_tuple=True).T

            xi = torch.cat(
                [box[i], xi[i, j + 4, None], j[:, None].float(), extra[i]], 1
            )

            if len(xi) == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue

            xi = xi[torch.argsort(xi[:, 4], descending=True)[:max_nms]]

            # NMS
            c = xi[:, 5:6] * max_wh
            boxes, scores = xi[:, :4] + c, xi[:, 4]
            i = torchvision.ops.nms(boxes, scores, self.iou_thres)
            i = i[:max_det]
            output.append(xi[i])

        return output


class YOLOAnchorlessSegPost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
