import numpy as np
import torch
from .base import YOLOPostBase


class YOLOAnchorPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
        self.make_anchor_grid()

    def make_anchor_grid(self):
        self.grid, self.anchor_grid, self.stride = [], [], []

        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]

        out_sizes = [[self.imh // strd, self.imw // strd] for strd in strides]
        for anchr, (ny, nx), strd in zip(self.anchors, out_sizes, strides):
            x, y = torch.arange(nx, dtype=torch.float32), torch.arange(
                ny, dtype=torch.float32
            )
            xv, yv = torch.meshgrid(x, y)
            grid = torch.broadcast_to(
                torch.stack((xv, yv), dim=-1).reshape(1, 1, ny, nx, 2),
                (1, self.na, ny, nx, 2),
            )
            self.grid.append(grid)

            anchr = torch.broadcast_to(
                torch.tensor(anchr).reshape(1, self.na, 1, 1, 2),
                (1, self.na, ny, nx, 2),
            )
            self.anchor_grid.append(anchr)
            self.stride.append(
                strd * torch.ones((1, self.na, ny, nx, 2), dtype=torch.float32)
            )

    def rearrange(self, x):
        return x

    def decode(self, x):
        return x

    def nms(self, x):
        return x


class YOLOAnchorSegPost(YOLOAnchorPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
