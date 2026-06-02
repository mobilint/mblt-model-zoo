from __future__ import annotations

from typing import Any, cast

import torch

from .base import YOLOPostBase
from .common import YOLOPosePostMixin, YOLOSegPostMixin, dist2bbox, dual_topk


class YOLODFLFreePost(YOLOPostBase):
    """Postprocessing for YOLO DFL-free models."""

    max_det = 300

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        """Initialize the DFL-free YOLO postprocessor.

        Args:
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Return the export-style output tensor for DFL-free YOLO models."""
        if len(x) == 2:
            converted = cast(torch.Tensor, self.conversion(x))
            return self._stack_topk_outputs(self.filter_conversion(converted))
        if len(x) == 4:
            converted, proto_outs = cast(tuple[torch.Tensor, torch.Tensor], self.conversion(x))
            return [self._stack_topk_outputs(self.filter_conversion(converted)), self._proto_to_nchw(proto_outs)]
        if len(x) == 3:
            converted = cast(torch.Tensor, self.conversion(x))
            return self._stack_topk_outputs(self.filter_conversion(converted))

        rearranged = self.rearrange(x)
        if isinstance(rearranged, tuple):
            det_out, proto_outs = rearranged
            return [self.decode_batch(det_out), self._proto_to_nchw(proto_outs)]
        return self.decode_batch(rearranged)

    def _proto_to_nchw(self, proto: torch.Tensor) -> torch.Tensor:
        """Convert prototype tensors to ``(B, C, H, W)`` if needed."""
        if proto.ndim == 4 and proto.shape[1] == self.n_extra:
            return proto
        if proto.ndim == 4 and proto.shape[-1] == self.n_extra:
            return proto.permute(0, 3, 1, 2)
        raise ValueError(f"Unsupported proto tensor shape {tuple(proto.shape)} for non-e2e output.")

    def _stack_topk_outputs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Pad or trim per-image detections to a fixed batch tensor."""
        output_dim = 6 + self.n_extra
        padded_outputs = []
        for output in outputs:
            output = output[: self.max_det]
            if output.shape[0] < self.max_det:
                pad = torch.zeros(
                    (self.max_det - output.shape[0], output_dim),
                    dtype=output.dtype,
                    device=output.device,
                )
                output = torch.cat([output, pad], dim=0)
            padded_outputs.append(output)
        return torch.stack(padded_outputs, dim=0)

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every anchor, then apply batched top-k selection for export-style output."""
        box, scores, extra = torch.split(x, [4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        dbox = dist2bbox(box, anchors, xywh=False, dim=1) * stride
        decoded = torch.cat([dbox, scores.sigmoid(), extra], dim=1).transpose(1, 2)
        return self._stack_topk_outputs(
            [
                dual_topk(image, self.nc, self.n_extra, max_det=self.max_det, conf_thres=self.conf_thres)
                for image in decoded
            ]
        )

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[Any, torch.Tensor | None]:
        """Preprocesses inputs for DFL-free models.

        Args:
            x (list[torch.Tensor]): Raw model outputs.

        Returns:
            tuple: (processed detections, None).
        """
        if len(x) == 2:
            converted = cast(torch.Tensor, self.conversion(x))
            return self.filter_conversion(converted), None
        rearranged = self.rearrange(x)
        if not isinstance(rearranged, torch.Tensor):
            raise TypeError("rearrange should return a tensor for DFL-free detection postprocessing.")
        return self.decode(rearranged), None

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Converts raw model output tensors into a single concatenated tensor.

        Args:
            x (list[torch.Tensor]): List of raw output tensors.

        Returns:
            torch.Tensor:
                Concatenated tensor of shape ``(batch, num_anchors, 4 + nc + n_extra)``.
        """
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=self.nc < 4)
        return torch.cat(x, dim=-1).squeeze(1)  # [b, 8400, 84]

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rearranges raw outputs into a task-specific intermediate representation.

        Args:
            x: Raw model output tensors.

        Returns:
            A concatenated intermediate representation used by ``decode``.
        """
        y_det = []
        y_cls = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")
            if xi.shape[-1] == 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 4, 80, 80), (b, 4, 40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(xi.permute(0, 3, 1, 2))  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        return torch.cat(
            [torch.cat((yi_det, yi_cls), dim=1).flatten(2) for yi_det, yi_cls in zip(y_det, y_cls)],
            dim=-1,
        )

    def decode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Decodes model outputs into box coordinates and class scores.

        Args:
            x (torch.Tensor): Concatenated output tensor from `rearrange`.

        Returns:
            list[torch.Tensor]: Per-image decoded detections after filtering and top-k selection.
        """
        return [self.process_box_cls(box_cls) for box_cls in x]

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        """Processes detection results for a single image.

        Args:
            box_cls: Raw detections for one image.

        Returns:
            Decoded boxes, scores, and extra data.
        """
        if self.n_extra == 0:
            ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.inv_conf_thres
        else:
            ic = torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]  # (84, *)
        if box_cls.numel() == 0:
            return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)  # (0, 84)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, extra = torch.split(box_cls[None], [4, self.nc, self.n_extra], dim=1)  # (*, 4), (*, 80), (*, 32)
        dbox = (
            dist2bbox(
                box,
                anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * stride[:, ic]
        )
        pre_topk = torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0).transpose(0, 1)  # (*, 84)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres)

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Output tensor from the model.

        Returns:
            list[torch.Tensor]: Filtered detections for each image in the batch.
        """
        x_list = torch.split(x, 1, dim=0)  # [(1, 8400, 84), (1, 8400, 84), ...]

        return [dual_topk(xi.squeeze(0), self.nc, self.n_extra, conf_thres=self.conf_thres) for xi in x_list]

    def nms(
        self,
        x: torch.Tensor | list[torch.Tensor],
        _max_det: int = 300,
        _max_nms: int = 30000,
        _max_wh: int = 7680,
    ) -> list[torch.Tensor]:
        """Performs Non-Maximum Suppression (no-op for NMS-free models).

        Args:
            x (list[torch.Tensor]): Decoded detections.
            _max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            _max_nms (int, optional): Maximum candidates for NMS. Defaults to 30000.
            _max_wh (int, optional): Maximum box width/height. Defaults to 7680.

        Returns:
            list[torch.Tensor]: Per-image detections with padded zero rows removed.
        """
        if isinstance(x, list):
            return x
        return [xi[xi[:, 4] > 0] for xi in x]


class YOLODFLFreeSegPost(YOLOSegPostMixin, YOLODFLFreePost):
    """Postprocessing for YOLO NMS-free segmentation models."""

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Preprocesses intermediate inputs into (boxes, proto) format.

        Args:
            x (list[torch.Tensor]): Raw model output tensors.

        Returns:
            tuple: (decoded_detections, prototype_masks).
        """
        if len(x) == 4:
            converted, proto_outs = cast(tuple[torch.Tensor, torch.Tensor], self.conversion(x))
            return self.filter_conversion(converted), proto_outs
        rearranged, proto_outs = self.rearrange(x)
        return self.decode(rearranged), proto_outs

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Converts raw outputs into detections and prototype masks.

        Args:
            x: Input tensors.

        Returns:
            A tuple of processed detections and prototype masks.
        """

        x = sorted(x, key=lambda x: x.size(), reverse=self.nc < 4)
        outputs: list[torch.Tensor] = []
        protos: list[torch.Tensor] = []
        for xi in x:
            if xi.shape[-1] == self.n_extra:
                protos.append(xi)
            else:
                outputs.append(xi)
        proto = protos.pop(0 if self.nc < 4 else -1)
        converted = torch.cat(outputs + protos, dim=-1).squeeze(1)
        return converted, proto

    def rearrange(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rearranges segmentation outputs into detections and prototype masks.

        Args:
            x: Raw model output tensors.

        Returns:
            A tuple of concatenated detections and prototype masks.
        """
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
            if xi.shape[-1] == self.n_extra:
                y_ext.append(xi.permute(0, 3, 1, 2))  # (b, 32, 160, 160), (b, 32, 80, 80), ...
            elif xi.shape[-1] == 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 4, 80, 80), (b, 4 ,40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(xi.permute(0, 3, 1, 2))  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_ext = sorted(y_ext, key=lambda x: x.numel(), reverse=True)
        proto = y_ext.pop(0).permute(0, 2, 3, 1)
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        assert len(y_cls) == len(y_det) == len(y_ext), "output arguments are not in a proper form"
        y = torch.cat(
            [
                torch.cat((yi_det, yi_cls, yi_ext), dim=1).flatten(2)
                for yi_det, yi_cls, yi_ext in zip(y_det, y_cls, y_ext)
            ],
            dim=-1,
        )
        return y, proto


class YOLODFLFreePosePost(YOLOPosePostMixin, YOLODFLFreePost):
    """Postprocessing for YOLO NMS-free pose estimation models."""

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Preprocesses inputs for pose estimation.

        Args:
            x (list[torch.Tensor]): Raw model outputs.

        Returns:
            tuple: (processed_detections, None).
        """
        if len(x) == 3:
            converted = cast(torch.Tensor, self.conversion(x))
            return self.filter_conversion(converted), None
        rearranged = self.rearrange(x)
        return self.decode(rearranged), None

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert input tensors.
        Args:
            x (list[torch.Tensor]): Input tensors.
        Returns:
            torch.Tensor: Converted tensor.
        """
        # sort by element number
        x = sorted(x, key=lambda x: x.size(), reverse=True)
        kpt: torch.Tensor = x.pop(0)
        kpt = kpt.permute(0, 3, 1, 2).flatten(-2)
        return torch.cat([torch.cat(x, dim=-1).squeeze(1), kpt], dim=-1)  # [b, 8400, 56]

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor:
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
            if xi.shape[-1] == 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 4, 80, 80), (b, 4 ,40, 40), ...
            elif xi.shape[-1] == self.nc:
                y_cls.append(xi.permute(0, 3, 1, 2))  # (b, 1, 80, 80), (b, 1, 40, 40), ...
            elif xi.shape[-1] == self.n_extra:
                y_kpt.append(xi.permute(0, 3, 1, 2).flatten(2))  # (b, 51, 80, 80), (b, 1, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")
        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y_kpt = sorted(y_kpt, key=lambda x: x.numel(), reverse=True)  # (b, 51, 6400), (b, 51, 1600), (b, 51, 400)
        y_tmp = [
            torch.cat((yi_det, yi_cls), dim=1).flatten(2)
            for (yi_det, yi_cls) in zip(y_det, y_cls)  # (b, 65, 6400), (b, 65, 1600), (b, 65, 400)
        ]
        return torch.cat([torch.cat((yi_tmp, yi_kpt), dim=1) for yi_tmp, yi_kpt in zip(y_tmp, y_kpt)], dim=-1)

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        """Processes pose estimation results for a single image.

        Args:
            box_cls: Raw detections for one image.

        Returns:
            Decoded boxes, scores, and keypoints.
        """
        ic = torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]  # (116, *)
        if box_cls.numel() == 0:
            return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)  # (0, 56)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, keypoints = torch.split(
            box_cls[None], [4, self.nc, self.n_extra], dim=1
        )  # (1, 4, *), (1, 1, *), (1, 51, *)
        dbox = (
            dist2bbox(
                box,
                anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * stride[:, ic]
        )
        keypoints = keypoints.view(1, 17, 3, -1)
        key_coord, key_conf = torch.split(keypoints, [2, 1], dim=2)  # (1, 17, 2, 8400), (1, 17, 1, 8400)
        key_coord = (key_coord + anchors[:, ic]) * stride[:, ic]  # (1, 17, 2, *)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(1, self.n_extra, -1)  # (1, 51, *)
        pre_topk = torch.cat([dbox, scores.sigmoid(), keypoints], dim=1).squeeze(0).transpose(0, 1)  # (*, 56)
        return dual_topk(pre_topk, self.nc, self.n_extra, conf_thres=self.conf_thres)

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every anchor, then apply batched top-k selection for export-style pose output."""
        box, scores, keypoints = torch.split(x, [4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        dbox = dist2bbox(box, anchors, xywh=False, dim=1) * stride
        keypoints = keypoints.view(x.shape[0], 17, 3, -1)
        key_coord, key_conf = torch.split(keypoints, [2, 1], dim=2)
        key_coord = (key_coord + anchors.unsqueeze(1)) * stride.unsqueeze(1)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(x.shape[0], self.n_extra, -1)
        decoded = torch.cat([dbox, scores.sigmoid(), keypoints], dim=1).transpose(1, 2)
        return self._stack_topk_outputs(
            [
                dual_topk(image, self.nc, self.n_extra, max_det=self.max_det, conf_thres=self.conf_thres)
                for image in decoded
            ]
        )
