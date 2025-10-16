import torch
import torch.nn.functional as F
from .base import YOLOPostBase
from .common import *


class YOLOAnchorlessPost(YOLOPostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
        self.make_anchors()
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor (144)
        self.dfl_weight = torch.arange(
            self.reg_max, dtype=torch.float, device=self.device
        ).reshape(1, -1, 1, 1)

    def dfl(self, x):
        assert x.ndim == 3, "Assume that x is a 3d tensor"
        b, _, a = x.shape
        return F.conv2d(
            x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1), self.dfl_weight
        ).view(b, 4, a)

    def process_extra(self, x, ic):
        return x

    def make_anchors(self, offset=0.5):
        anchor_points, stride_tensor = [], []
        strides = [2 ** (3 + i) for i in range(self.nl)]
        if self.nl == 2:
            strides = [strd * 2 for strd in strides]

        for strd in strides:
            ny, nx = self.imh // strd, self.imw // strd
            sy = torch.arange(ny, dtype=torch.float32, device=self.device) + offset
            sx = torch.arange(nx, dtype=torch.float32, device=self.device) + offset
            yv, xv = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((xv, yv), -1).reshape(-1, 2))
            stride_tensor.append(
                torch.full((ny * nx, 1), strd, dtype=torch.float32, device=self.device)
            )

        self.anchors = torch.cat(anchor_points, dim=0).permute(1, 0)
        self.stride = torch.cat(stride_tensor, dim=0).permute(1, 0)

    def rearrange(self, x):
        y_det = []
        y_cls = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)

            assert xi.ndim == 4, f"Got unexpected shape for x={x.shape}."

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)

        y = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(y_det, y_cls)
        ]

        return y

    def decode(self, x):
        batch_box_cls = torch.cat(x, axis=-1)  # (b, no=144, 8400)

        y = []
        for xi in batch_box_cls:
            if self.n_extra == 0:
                ic = (torch.amax(xi[-self.nc :, :], dim=0) > self.inv_conf_thres).to(
                    self.device
                )

            else:
                ic = (
                    torch.amax(xi[-self.nc - self.n_extra : -self.n_extra, :], dim=0)
                    > self.inv_conf_thres
                ).to(self.device)

            xi = xi[:, ic]  # (144, *)

            if xi.numel() == 0:
                y.append(
                    torch.zeros(
                        (0, 4 + self.nc + self.n_extra),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                continue

            box, score, extra = torch.split(
                xi[None], [self.reg_max * 4, self.nc, self.n_extra], 1
            )  # (1, 64, *), (1, nc, *), (1, n_extra, *)
            dbox = (
                dist2bbox(self.dfl(box), self.anchors[:, ic], xywh=False, dim=1)
                * self.stride[:, ic]
            )
            extra = self.process_extra(extra, ic)
            y.append(
                torch.cat([dbox, score.sigmoid(), extra], dim=1)
                .squeeze(0)
                .transpose(1, 0)
            )

        return y

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
        x, proto_outs = self.rearrange(x)
        x = self.decode(x)
        x = self.nms(x)
        return self.masking(x, proto_outs)

    def rearrange(self, x):
        y_det = []
        y_cls = []
        y_ext = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")

            if xi.shape[1] == self.n_extra:
                y_ext.append(xi)  # (b, 32, 160, 160), (b, 32, 80, 80), ...
            elif xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_ext = sorted(y_ext, key=lambda x: x.numel(), reverse=True)
        proto = y_ext.pop(0)
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)

        assert (
            len(y_cls) == len(y_det) == len(y_ext)
        ), "output arguments are not in a proper form"

        y = [
            torch.cat((yi_det, yi_cls, yi_ext), dim=1).flatten(2)
            for (yi_det, yi_cls, yi_ext) in zip(y_det, y_cls, y_ext)
        ]

        return y, proto


class YOLOAnchorlessPosePost(YOLOAnchorlessPost):
    def __init__(
        self,
        pre_cfg: dict,
        post_cfg: dict,
    ):
        super().__init__(pre_cfg, post_cfg)

    def rearrange(self, x):
        y_det = []
        y_cls = []
        y_kpt = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 1, 80, 80), (b, 1, 40, 40), ...
            elif xi.shape[1] == self.n_extra:
                y_kpt.append(xi.flatten(2))  # (b, 51, 80, 80), (b, 1, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y_kpt = sorted(
            y_kpt, key=lambda x: x.numel(), reverse=True
        )  # (b, 51, 6400), (b, 51, 1600), (b, 51, 400)

        y_tmp = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(
                y_det, y_cls
            )  # (b, 65, 6400), (b, 65, 1600), (b, 65, 400)
        ]
        y = [
            torch.cat((yi_tmp, yi_kpt), dim=1) for (yi_tmp, yi_kpt) in zip(y_tmp, y_kpt)
        ]  # (b, 116, 6400), (b, 116, 1600), (b, 116, 400)

        return y

    def process_extra(self, kpt, ic):
        kpt = kpt.squeeze(0)
        assert kpt.shape[0] == self.n_extra, "keypoint shape mismatch"

        kpt = kpt.reshape(17, 3, -1)
        coord, conf = torch.split(kpt, [2, 1], dim=1)  # (17, 2, *), (17, 1, *)

        coord = (coord * 2 + (self.anchors[:, ic] - 0.5)) * self.stride[:, ic]
        conf = conf.sigmoid()
        kpt = torch.cat([coord, conf], dim=1).reshape(self.n_extra, -1)
        return kpt.unsqueeze(0)
