import torch
import torchvision
from .base import YOLOPostBase
from .common import *


class YOLOAnchorPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
        self.no = self.nc + 5 + self.n_extra
        self.make_anchor_grid()

    def make_anchor_grid(self):
        self.grid, self.anchor_grid, self.stride = [], [], []

        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]

        out_sizes = [
            [self.imh // strd, self.imw // strd] for strd in strides
        ]  # (80, 80), (40, 40), (20, 20)
        for anchr, (ny, nx), strd in zip(self.anchors, out_sizes, strides):
            yv, xv = torch.meshgrid(
                torch.arange(ny, dtype=torch.float32),
                torch.arange(nx, dtype=torch.float32),
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

    def rearrange(self, x):
        y = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            assert xi.ndim == 4, f"Got unexpected shape for x={x.shape}."

            if xi.shape[1] == self.no * self.na:
                y.append(xi)
            elif xi.shape[3] == self.no * self.na:
                y.append(xi.permute(0, 3, 1, 2))
            else:
                raise ValueError(f"Got unexpected shape for x={x.shape}.")

        y = sorted(
            y, key=lambda x: x.numel(), reverse=True
        )  # sort by number of elements in descending order
        return y

    def decode(self, x):
        x = torch.cat(
            [
                xi.reshape(xi.shape[0], self.na, self.no, xi.shape[2], xi.shape[3])
                .permute(0, 1, 3, 4, 2)
                .reshape(xi.shape[0], -1, self.no)
                for xi in x
            ],
            dim=1,
        )
        x = [xi for xi in x]  # convert to list of tensors if shape is (25200, 85)

        y = []
        for xi in x:
            ic = xi[:, 4] > self.conf_thres  # candidate indices
            box_cls = xi[ic]  # candidate boxes
            if len(box_cls) == 0:
                y.append(torch.zeros((0, self.no), dtype=torch.float32))
                continue
            grid = self.grid[ic]
            anchor_grid = self.anchor_grid[ic]
            stride = self.stride[ic]

            xy, wh, conf, scores, extra = torch.split(
                box_cls, [2, 2, 1, self.nc, self.n_extra], dim=-1
            )

            xy = (xy * 2.0 - 0.5 + grid) * stride
            wh = (wh * 2) ** 2 * anchor_grid
            y.append(torch.cat([xy, wh, conf, scores, extra], dim=-1))
        return y

    def nms(self, x, max_det=300, max_nms=30000, max_wh=7680):
        mi = 5 + self.nc  # mask index
        output = []
        for xi in x:
            if len(xi) == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue
            xi[..., 5:] *= xi[..., 4:5]  # conf = obj_conf * cls_conf

            box = xywh2xyxy(xi[..., :4])
            mask = xi[..., mi:]

            i, j = (xi[..., 5:mi] > self.conf_thres).nonzero(as_tuple=False).T
            xi = torch.cat([box[i], xi[i, j + 5, None], j[:, None].float(), mask[i]], 1)

            if len(xi) == 0:
                output.append(torch.zeros((0, 6 + self.n_extra), dtype=torch.float32))
                continue

            xi = xi[xi[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # NMS
            c = xi[:, 5:6] * max_wh
            boxes, scores = xi[:, :4] + c, xi[:, 4]
            i = torchvision.ops.nms(boxes, scores, self.iou_thres)[:max_det]
            output.append(xi[i])

        return output


class YOLOAnchorSegPost(YOLOAnchorPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
