import torch
from .yolo_anchorless_post import YOLOAnchorlessPost
from .common import *


class YOLONMSFreePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)
        assert (
            self.n_extra == 0
        ), "YOLOv10 is not implemented for segmentation, pose estimation"

    def decode(self, x):
        batch_box_cls = torch.cat(x, axis=-1)  # (b, no=144, 8400)

        y = []
        for xi in batch_box_cls:
            ic = (torch.amax(xi[-self.nc :, :], dim=0) > self.inv_conf_thres).to(
                self.device
            )

            xi = xi[:, ic]  # (144, *)

            if xi.numel() == 0:
                y.append(
                    torch.zeros(
                        (0, 6),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                continue

            box, score = torch.split(
                xi[None], [self.reg_max * 4, self.nc], 1
            )  # (1, 64, *), (1, nc, *)
            dbox = (
                dist2bbox(self.dfl(box), self.anchors[:, ic], xywh=False, dim=1)
                * self.stride[:, ic]
            )
            pre_topk = (
                torch.cat([dbox, score.sigmoid()], dim=1).squeeze(0).transpose(0, 1)
            )  # (*, 84)

            max_det = min(pre_topk.shape[0], 300)

            box, score = pre_topk.split([4, self.nc], dim=-1)
            max_score = score.amax(dim=-1)
            max_score, index = torch.topk(
                max_score, max_det, dim=-1
            )  # max deteciton is 300
            index = index.unsqueeze(-1)
            box = torch.gather(box, dim=0, index=index.repeat(1, 4))
            score = torch.gather(score, dim=0, index=index.repeat(1, self.nc))

            # second topk
            score, index = torch.topk(score.flatten(), max_det)
            index = index.unsqueeze(-1)
            score = score.unsqueeze(-1)
            label = index % self.nc
            index = index // self.nc
            box = box.gather(dim=0, index=index.repeat(1, 4))

            box_cls = torch.cat([box, score, label], dim=1)  # (300, 6)
            box_cls = box_cls[box_cls[:, 4] > self.conf_thres]  # final filtering

            if box_cls.numel() == 0:
                y.append(
                    torch.zeros(
                        (0, 6),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                continue
            y.append(box_cls)

        return y

    def nms(self, x):  # Do nothing on NMS Free model
        return x
