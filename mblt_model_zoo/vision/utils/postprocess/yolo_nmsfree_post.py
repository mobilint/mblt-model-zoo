"""
YOLO NMS-free postprocessing.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from .common import dist2bbox, dual_topk
from .yolo_anchorless_post import YOLOAnchorlessPost


class YOLONMSFreePost(YOLOAnchorlessPost):
    """Postprocessing for YOLO NMS-free models."""

    max_det = 300

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return the export-style output tensor for NMS-free YOLO models."""
        if len(x) == 2:
            converted = cast(torch.Tensor, self.conversion(x))
            return self._stack_topk_outputs(self.filter_conversion(converted))

        rearranged = cast(torch.Tensor, self.rearrange(x))
        return self.decode_batch(rearranged)

    def _stack_topk_outputs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Pad or trim per-image detections to a fixed batch tensor."""
        padded_outputs = []
        for output in outputs:
            output = output[: self.max_det]
            if output.shape[0] < self.max_det:
                pad = torch.zeros(
                    (self.max_det - output.shape[0], 6),
                    dtype=output.dtype,
                    device=output.device,
                )
                output = torch.cat([output, pad], dim=0)
            padded_outputs.append(output)
        return torch.stack(padded_outputs, dim=0)

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every anchor, then apply batched top-k selection for export-style output."""
        box, scores = torch.split(x, [self.reg_max * 4, self.nc], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        dbox = dist2bbox(self.dfl(box), anchors, xywh=False, dim=1) * stride
        decoded = torch.cat([dbox, scores], dim=1).transpose(1, 2)
        return self._stack_topk_outputs(
            [
                dual_topk(
                    image,
                    self.nc,
                    self.n_extra,
                    max_det=self.max_det,
                    conf_thres=self.conf_thres,
                    score_is_logits=True,
                )
                for image in decoded
            ]
        )

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[Any, torch.Tensor | None]:
        """Preprocesses inputs for NMS-free models.

        Args:
            x (list[torch.Tensor]): Raw model outputs.

        Returns:
            tuple: (processed_detections, None).
        """
        if len(x) == 2:
            converted = cast(torch.Tensor, self.conversion(x))
            return self.filter_conversion(converted), None
        rearranged = cast(torch.Tensor, self.rearrange(x))
        return self.decode(rearranged), None

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert input tensors.
        Args:
            x (list[torch.Tensor]): Input tensors.
        Returns:
            torch.Tensor: Converted tensor.
        """
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=self.nc < 4)
        return torch.cat(x, dim=-1).squeeze(1)  # [b, 8400, 84]

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters out low-confidence detections from a single output tensor.

        Args:
            x (torch.Tensor): Model output tensor.

        Returns:
            list[torch.Tensor]: Decoded and filtered outputs for each image.
        """
        x_list = torch.split(x, 1, dim=0)  # [(1, 8400, 84), (1, 8400, 84), ...]

        return [dual_topk(xi.squeeze(0), self.nc, self.n_extra, conf_thres=self.conf_thres) for xi in x_list]

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        """Processes detection results for a single image.

        Args:
            box_cls: Raw detections for one image.

        Returns:
            Decoded and top-k filtered detections.
        """
        ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros((0, 6), dtype=torch.float32)  # (0, 6)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores = torch.split(box_cls[None], [self.reg_max * 4, self.nc], dim=1)  # (1, 64, *), (1, 80, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * stride[:, ic]
        )
        pre_topk = torch.cat([dbox, scores], dim=1).squeeze(0).transpose(0, 1)  # (*, 84)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres, score_is_logits=True)

    def nms(
        self,
        x: torch.Tensor | list[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> list[torch.Tensor]:
        """Perform Non-Maximum Suppression (no-op for NMS-free models).

        Args:
            x (list[torch.Tensor]): Decoded detections.
            max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            max_nms (int, optional): Maximum candidates for NMS. Defaults to 30000.
            max_wh (int, optional): Maximum box width/height. Defaults to 7680.

        Returns:
            list[torch.Tensor]: Per-image detections with padded zero rows removed.
        """
        if isinstance(x, list):
            return x
        return [xi[xi[:, 4] > 0] for xi in x]
