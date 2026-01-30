from typing import List

import torch

from .yolo_anchorless_post import YOLOAnchorlessPost


class YOLONMSFreePost(YOLOAnchorlessPost):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__(pre_cfg, post_cfg)

    def conversion(self, x: List[torch.Tensor]):
        assert (
            len(x) == 2
        ), f"Assume return is a list of two outputs, but got {len(x)} outputs"
        return torch.concat(x, dim=-1)  # [b, 8400, 84]

    def filter_conversion(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convert the output of ONNX model to the input of NMS.

        Args:
            x: NPU outputs
        Returns:
            Decoded outputs
        """
        x_list = torch.split(x, 1, dim=0)  # [(1, 8400, 84), (1, 8400, 84), ...]

        def process_conversion(x):
            x = x.squeeze(0)

            if self.n_extra == 0:
                ic = torch.amax(x[..., -self.nc :], dim=-1) > self.conf_thres
            else:
                ic = (
                    torch.amax(x[..., -self.nc - self.n_extra : -self.n_extra], dim=-1)
                    > self.conf_thres
                )
            pre_topk = x[ic]  # (*, 84)
            max_det = min(pre_topk.shape[0], 300)

            # first topk
            box, scores, extra = pre_topk.split([4, self.nc, self.n_extra], dim=-1)
            max_scores = scores.amax(dim=-1)
            max_scores, index = torch.topk(
                max_scores, max_det, dim=-1
            )  # max deteciton is 300
            index = index.unsqueeze(-1)
            box = torch.gather(box, dim=0, index=index.repeat(1, 4))
            scores = torch.gather(scores, dim=0, index=index.repeat(1, self.nc))
            extra = torch.gather(extra, dim=0, index=index.repeat(1, self.n_extra))

            # second topk
            scores, index = torch.topk(scores.flatten(), max_det)
            index = index.unsqueeze(-1)
            scores = scores.unsqueeze(-1)
            labels = index % self.nc
            index = index // self.nc
            box = box.gather(dim=0, index=index.repeat(1, 4))
            extra = extra.gather(dim=0, index=index.repeat(1, self.n_extra))

            box_cls = torch.cat([box, scores, labels, extra], dim=1)  # (300, 6)

            box_cls = box_cls[box_cls[:, 4] > self.conf_thres]  # final filtering
            if box_cls.numel() == 0:
                return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32)

            return box_cls

        return [process_conversion(xi) for xi in x_list]

    def nms(self, x):  # Do nothing on NMS Free model
        return x
