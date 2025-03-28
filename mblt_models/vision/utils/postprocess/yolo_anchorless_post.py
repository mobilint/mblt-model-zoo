from .base import YOLOPostBase
from .common import *


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
        self.stride = [2 ** (3 + i) for i in range(self.nl)]
        self.make_anchors()
        self.reg_max = 16  # DFL channels
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor (144)
        self.dfl = DFL(self.reg_max)

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
        return x

    def decode(self, x):
        return x

    def nms(self, x):
        return x


class YOLOAnchorlessSegPost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict, conf_thres=0.001, iou_thres=0.7):
        super().__init__(pre_cfg, post_cfg, conf_thres, iou_thres)
