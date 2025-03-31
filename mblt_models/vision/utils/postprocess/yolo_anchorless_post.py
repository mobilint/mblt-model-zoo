import torch
import torchvision
from .base import YOLOPostBase
from .common import *


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
        self.make_anchors()

    def make_anchors(self, offset=0.5):
        anchor_points, stride_tensor = [], []
        strides = [2 ** (3 + i) for i in range(self.nl)]
        for strd in strides:
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
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            assert xi.ndim == 4, f"Got unexpected shape for x={x.shape}."
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

    def __call__(self, x):
        x = self.check_input(x)
        x, proto_outs = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def rearrange(self, x):
        # for xi in x, unsqueeze the first dimension if ndim(xi) == 3
        for i, xi in enumerate(x):
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
                x[i] = xi

        x = sorted(x, key=lambda x: x.numel(), reverse=True)  # sort by numel
        proto = x.pop(0)  # (b, 32, 160, 160)
        y = []
        for xi in x:
            if xi.shape[1] == 1:  # coord
                xi = xi.squeeze(1) * self.stride
                y.append(xi)
            elif xi.shape[1] == self.nc:  # cls
                y.append(xi.squeeze(2))
            elif xi.shape[1] == self.n_extra:  # mask
                y.append(xi.squeeze(2))
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        y = [y[-1]] + y[:-1]  # move mask to the front
        y = torch.cat(y, dim=1)  # (b, 116, 8400)

        return y, proto


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)

    def rearrange(self, x):
        y = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            assert xi.ndim == 4, f"Got unexpected shape for x={x.shape}."

            if xi.shape[1] == 1:  # cls
                xi = xi.squeeze(1)
                if xi.shape[1] == 4:  # coord
                    xi = xi * self.stride
                y.append(xi)
            elif xi.shape[3] == 3:  # keypoints
                xi = xi.reshape(xi.shape[0], -1, self.n_extra)
                xi = xi.transpose(0, 2, 1)
                y.append(xi)
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        y = sorted(
            y, key=lambda x: x.size, reverse=True
        )  # sort by size descending (bs, 51, 8400), (bs, 4, 8400), (bs, 1, 8400)
        return torch.cat(y, dim=1)  # (bs, 56, 8400)
