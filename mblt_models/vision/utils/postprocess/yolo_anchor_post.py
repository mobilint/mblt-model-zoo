import numpy as np
import torch
from .base import YOLOPostBase


def make_anchor_grids(nl, imh, imw, anchors):
    if nl not in [2, 3, 4]:
        raise ValueError(f"Your model has wrong number of detection layers: {nl}")
    na = len(anchors[0]) // 2  # number of anchors

    strides = [2 ** (3 + i) for i in range(nl)]  # strides [8, 16, 32] for YOLOv5
    if nl == 2:  # YOLOv3 tiny has 2 detection layers with strides 16, 32
        strides = [strd * 2 for strd in strides]
    out_sizes = [
        [imh // strd, imw // strd] for strd in strides
    ]  # output sizes [[80, 80], [40, 40], [20, 20]] for YOLOv5

    all_grids, all_anchor_grids, all_strides = [], [], []
    for anchr, (ny, nx), strd in zip(anchors, out_sizes, strides):
        x, y = np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        grid = np.broadcast_to(
            np.stack((xv, yv), axis=-1).reshape(1, 1, ny, nx, 2), (1, na, ny, nx, 2)
        )
        all_grids.append(grid)

        anchr = np.broadcast_to(
            np.array(anchr).reshape(1, na, 1, 1, 2), (1, na, ny, nx, 2)
        )
        all_anchor_grids.append(anchr)
        all_strides.append(strd * np.ones((1, na, ny, nx, 2), dtype=np.float32))

    return all_grids, all_anchor_grids, all_strides


def make_anchor_grid(nl, imh, imw, anchors):
    pass


class YOLOAnchorPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

    def rearrange(self, x):
        return x

    def decode(self, x):
        return x

    def nms(self, x):
        return x


class YOLOAnchorSegPost(YOLOAnchorPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

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
