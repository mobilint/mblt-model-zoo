"""
YOLO anchorless postprocessing.
"""

from __future__ import annotations

import torch

from .base import YOLOPostBase
from .common import (
    YOLOOBBPostMixin,
    YOLOPosePostMixin,
    YOLOSegPostMixin,
    concat_converted_obb_outputs,
    decode_split_converted_obb_outputs,
    dist2bbox,
    dist2rbox,
    non_max_suppression,
    rotated_nms,
    xywh2xyxy,
    yolo_multilabel_candidates,
)


class YOLOAnchorlessPost(YOLOPostBase):
    """Postprocessing for YOLO models without anchors."""

    def __init__(self, pre_cfg: dict, post_cfg: dict, **kwargs: object) -> None:
        """Initialize the anchorless YOLO postprocessor.

        Args:
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            **kwargs: Optional runtime overrides for postprocess behavior.
        """
        super().__init__(pre_cfg, post_cfg, **kwargs)
        self.reg_max = post_cfg.get("reg_max", 0)  # DFL channels
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor (144)
        self.dfl_weight = torch.arange(self.reg_max, dtype=torch.float32, device=self.device).reshape(1, -1, 1, 1)

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Return the export-style output tensor for anchorless YOLO models."""
        if len(x) == 1:
            converted = self.conversion(x)
            if isinstance(converted, torch.Tensor):
                return self._converted_to_batch_output(converted)
            det_out, proto_out = converted
            return [self._converted_to_batch_output(det_out), proto_out]

        rearranged = self.rearrange(x)
        if isinstance(rearranged, tuple):
            det_out, proto_out = rearranged
            return [self.decode_batch(det_out), proto_out.permute(0, 3, 1, 2)]
        return self.decode_batch(rearranged)

    def _converted_to_batch_output(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize converted outputs to the export-style batched layout."""
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D converted tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] == 4 + self.nc + self.n_extra:
            return x
        if x.shape[-1] == 4 + self.nc + self.n_extra:
            return x.transpose(1, 2)
        raise ValueError(f"Unsupported converted tensor shape {tuple(x.shape)} for non-e2e output.")

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every anchor without confidence filtering for export-style output."""
        box, scores, extra = torch.split(x, [self.reg_max * 4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        dbox = dist2bbox(self.dfl(box), anchors, xywh=False, dim=1) * stride
        return torch.cat([dbox, scores.sigmoid(), extra], dim=1)

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rearranges raw model output tensors into a concatenated decode input.

        Args:
            x (list[torch.Tensor]): List of raw output tensors from the model detection heads.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Concatenated tensor in
                ``(batch, channels, anchors)`` format, optionally paired with prototype masks.
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
            if xi.shape[-1] == self.reg_max * 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
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
            list[torch.Tensor]: Per-image decoded detections in ``(channels, anchors)`` format.
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
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros((4 + self.nc + self.n_extra, 0), dtype=torch.float32)  # (84, 0)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, extra = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 80, *), (1, 32, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * stride[:, ic]
        )
        return torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0)

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters out low-confidence detections from a single concatenated output tensor.

        Args:
            x (torch.Tensor): Concatenated output tensor from the model.

        Returns:
            list[torch.Tensor]: Filtered detections for each image in the batch.
        """
        while x.ndim == 4 and 1 in (x.shape[0], x.shape[1]):
            if x.shape[0] == 1:
                x = x.squeeze(0)
            elif x.shape[1] == 1:
                x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D converted tensor, got shape {tuple(x.shape)}.")
        expected_dim = 4 + self.nc + self.n_extra
        if x.shape[-1] == expected_dim:
            normalized = x
        elif x.shape[1] == expected_dim:
            normalized = x.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported converted tensor shape {tuple(x.shape)}.")
        x_list = torch.split(normalized, 1, dim=0)  # [(1, 8400, 84), (1, 8400, 84), ...]

        def process_conversion(x: torch.Tensor) -> torch.Tensor:
            x = x.squeeze(0)  # (8400, 84)
            if self.n_extra == 0:
                ic = torch.amax(x[:, -self.nc :], dim=1) > self.conf_thres
            else:
                ic = torch.amax(x[:, -self.nc - self.n_extra : -self.n_extra], dim=1) > self.conf_thres
            x = x[ic]
            if x.numel() == 0:
                return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)
            x = xywh2xyxy(x)
            return x

        return [process_conversion(xi) for xi in x_list]

    def _nms_single(self, xi: torch.Tensor, max_det: int, max_nms: int, max_wh: int) -> torch.Tensor:
        """Apply anchorless NMS to a single decoded image tensor."""
        if xi.numel() == 0:
            return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32, device=self.device)
        xi_t = xi.transpose(0, 1)
        score = xi_t[:, 4 : 4 + self.nc]
        extra = xi_t[:, 4 + self.nc :]
        conf, cls_idx = score.max(dim=1)
        filt = conf > self.conf_thres
        if not torch.any(filt):
            return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32, device=self.device)
        xi_t = xi_t[filt]
        conf = conf[filt]
        cls_idx = cls_idx[filt]
        extra = extra[filt]
        xi_out = torch.empty((xi_t.shape[0], 6 + self.n_extra), dtype=xi_t.dtype, device=xi_t.device)
        xi_out[:, :4] = xi_t[:, :4]
        xi_out[:, 4] = conf
        xi_out[:, 5] = cls_idx.to(xi_t.dtype)
        if self.n_extra > 0:
            xi_out[:, 6:] = extra
        xi_out = xi_out[torch.argsort(xi_out[:, 4], descending=True)[:max_nms]]
        c = xi_out[:, 5:6] * max_wh
        boxes, scores = xi_out[:, :4] + c, xi_out[:, 4]
        i_idx = non_max_suppression(boxes, scores, self.iou_thres, max_det)
        return xi_out[i_idx]

    def _nms_single_legacy_rows(self, xi: torch.Tensor, max_det: int, max_nms: int, max_wh: int) -> torch.Tensor:
        """Apply anchorless NMS to a single decoded image in row-major ``(anchors, channels)`` form."""
        if xi.numel() == 0:
            return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32, device=self.device)
        box, score, extra = xi[:, :4], xi[:, 4 : 4 + self.nc], xi[:, 4 + self.nc :]
        conf, cls_idx = score.max(dim=1)
        filt = conf > self.conf_thres
        if not torch.any(filt):
            return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32, device=self.device)
        box = box[filt]
        conf = conf[filt]
        cls_idx = cls_idx[filt]
        extra = extra[filt]
        xi_out = torch.empty((box.shape[0], 6 + self.n_extra), dtype=xi.dtype, device=xi.device)
        xi_out[:, :4] = box
        xi_out[:, 4] = conf
        xi_out[:, 5] = cls_idx.to(xi.dtype)
        if self.n_extra > 0:
            xi_out[:, 6:] = extra
        xi_out = xi_out[torch.argsort(xi_out[:, 4], descending=True)[:max_nms]]
        c = xi_out[:, 5:6] * max_wh
        boxes, scores = xi_out[:, :4] + c, xi_out[:, 4]
        i_idx = non_max_suppression(boxes, scores, self.iou_thres, max_det)
        return xi_out[i_idx]

    def nms(
        self,
        x: torch.Tensor | list[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> list[torch.Tensor]:
        """Performs Non-Maximum Suppression (NMS) on the decoded detections.

        Args:
            x (list[torch.Tensor]): Decoded detections for each image.
            max_det (int, optional): Maximum number of detections to keep. Defaults to 300.
            max_nms (int, optional): Maximum number of candidates to consider for NMS.
                Defaults to 30000.
            max_wh (int, optional): Maximum box width/height for offset calculation.
                Defaults to 7680.

        Returns:
            list[torch.Tensor]: Post-NMS detections for each image.
        """
        if isinstance(x, list):
            output = []
            for xi in x:
                if xi.ndim != 2:
                    raise ValueError(f"Expected 2D decoded tensor, got shape {tuple(xi.shape)}.")
                if xi.shape[1] == 4 + self.nc + self.n_extra:
                    output.append(self._nms_single_legacy_rows(xi, max_det=max_det, max_nms=max_nms, max_wh=max_wh))
                elif xi.shape[0] == 4 + self.nc + self.n_extra:
                    output.append(self._nms_single(xi, max_det=max_det, max_nms=max_nms, max_wh=max_wh))
                else:
                    raise ValueError(f"Unsupported decoded tensor shape {tuple(xi.shape)}.")
            return output
        return [self._nms_single(xi, max_det=max_det, max_nms=max_nms, max_wh=max_wh) for xi in x]

    def dfl(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Distribution Focal Loss projection.

        Args:
            x: Tensor with shape ``(B, 4 * reg_max, A)``.

        Returns:
            Tensor with shape ``(B, 4, A)`` containing projected distances.
        """
        if self.reg_max == 0:  # skip dfl for yolov6 s, n models
            return x
        assert x.ndim == 3, "Input must be a 3D tensor (B, 4 * reg_max, A)"
        B, _, A = x.shape
        x = x.view(B, 4, self.reg_max, A).softmax(dim=2)
        return (x * self.dfl_weight.view(1, 1, self.reg_max, 1)).sum(dim=2)


class YOLOAnchorlessSegPost(YOLOSegPostMixin, YOLOAnchorlessPost):
    """Postprocessing for YOLO segmentation models without anchors."""

    def non_e2e(self, x: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Return the export-style output tensor for anchorless YOLO segmentation models.

        Args:
            x: Checked raw model outputs.

        Returns:
            A detection tensor, or detections paired with prototype masks.
        """
        if len(x) == 2:
            converted, proto_outs = self.conversion(x)
            return [self._converted_to_batch_output(converted), self._proto_to_nchw(proto_outs)]
        return super().non_e2e(x)

    def _proto_to_nchw(self, proto: torch.Tensor) -> torch.Tensor:
        """Convert prototype tensors to ``(B, C, H, W)`` if needed.

        Args:
            proto: Prototype tensor from a model runtime.

        Returns:
            Prototype tensor in channel-first batch layout.
        """
        if proto.ndim == 4 and proto.shape[1] == self.n_extra:
            return proto
        if proto.ndim == 4 and proto.shape[-1] == self.n_extra:
            return proto.permute(0, 3, 1, 2)
        raise ValueError(f"Unsupported proto tensor shape {tuple(proto.shape)} for non-e2e output.")

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Preprocesses intermediate inputs into (boxes, proto) format.

        Args:
            x (list[torch.Tensor]): Raw model output tensors.

        Returns:
            tuple: (decoded_detections, prototype_masks).
        """
        if len(x) == 2:
            converted, proto_outs = self.conversion(x)
            return self.filter_conversion(converted), proto_outs
        rearranged, proto_outs = self.rearrange(x)
        return self.decode(rearranged), proto_outs

    def conversion(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts raw model output tensors into detections and prototypes.

        Args:
            x (list[torch.Tensor]): List of raw output tensors.

        Returns:
            tuple: (detections, prototypes)
        """
        if (self.nc + self.n_extra + 4) in x[0].shape[1:] and self.n_extra in x[1].shape[1:]:
            return (
                x[0],
                x[1],
            )
        if (self.nc + self.n_extra + 4) in x[1].shape[1:] and self.n_extra in x[0].shape[1:]:
            return (
                x[1],
                x[0],
            )
        raise ValueError(f"Wrong shape of input: {x[0].shape}, {x[1].shape}")

    def rearrange(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rearrange model output tensors for segmentation tasks.
        Args:
            x (list[torch.Tensor]): Raw output tensors.
        Returns:
            tuple: (concatenated_detections, prototype_masks)
        """
        y_det: list[torch.Tensor] = []
        y_cls: list[torch.Tensor] = []
        y_ext: list[torch.Tensor] = []
        for xi in x:
            if xi.shape[-1] == self.n_extra:
                y_ext.append(xi.permute(0, 3, 1, 2))  # (b, 32, 160, 160), (b, 32, 80, 80), ...
            elif xi.shape[-1] == self.reg_max * 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
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


class YOLOAnchorlessPosePost(YOLOPosePostMixin, YOLOAnchorlessPost):
    """Postprocessing for YOLO pose estimation models without anchors."""

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Rearranges model output tensors for pose estimation tasks.

        Args:
            x (list[torch.Tensor]): Raw output tensors.

        Returns:
            torch.Tensor: Concatenated tensor for decode.
        """
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
            if xi.shape[-1] == self.reg_max * 4:
                y_det.append(xi.permute(0, 3, 1, 2))  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
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
            return torch.zeros((4 + self.nc + self.n_extra, 0), dtype=torch.float32)  # (56, 0)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, keypoints = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 1, *), (1, 51, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * stride[:, ic]
        )
        keypoints = keypoints.view(1, 17, 3, -1)
        key_coord, key_conf = torch.split(keypoints, [2, 1], dim=2)  # (1, 17, 2, 8400), (1, 17, 1, 8400)
        key_coord = (key_coord * 2 + anchors[:, ic] - 0.5) * stride[:, ic]  # (1, 17, 2, *)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(1, self.n_extra, -1)  # (1, 51, *)
        return torch.cat([dbox, scores.sigmoid(), keypoints], dim=1).squeeze(0)  # (56, *)

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every anchor without confidence filtering for export-style pose output."""
        box, scores, keypoints = torch.split(x, [self.reg_max * 4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        dbox = dist2bbox(self.dfl(box), anchors, xywh=False, dim=1) * stride
        keypoints = keypoints.view(x.shape[0], 17, 3, -1)
        key_coord, key_conf = torch.split(keypoints, [2, 1], dim=2)
        key_coord = (key_coord * 2 + anchors.unsqueeze(1) - 0.5) * stride.unsqueeze(1)
        keypoints = torch.cat([key_coord, key_conf.sigmoid()], dim=2).view(x.shape[0], self.n_extra, -1)
        return torch.cat([dbox, scores.sigmoid(), keypoints], dim=1)


class YOLOAnchorlessOBBPost(YOLOOBBPostMixin, YOLOAnchorlessPost):
    """Postprocessing for anchorless YOLO OBB models."""

    def _angle_from_raw(self, angle: torch.Tensor) -> torch.Tensor:
        """Decode YOLOv8/YOLO11 raw angle logits to radians."""
        return (angle.sigmoid() - 0.25) * torch.pi

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Preprocess OBB inputs into row-major detections.

        Args:
            x: Raw model outputs.

        Returns:
            A tuple of detections and no prototype output.
        """
        if len(x) in {1, 3, 5}:
            return self.filter_conversion(self.conversion(x)), None
        return self.decode(self.rearrange(x)), None

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Convert exported OBB outputs to one canonical tensor.

        Args:
            x: Converted OBB runtime outputs.

        Returns:
            Detections in ``cx, cy, w, h, class scores..., angle`` format.
        """
        if len(x) == 5:
            return decode_split_converted_obb_outputs(
                x,
                self.nc,
                self.n_extra,
                self.anchors_as_tensor(),
                self.stride_as_tensor(),
            )
        return concat_converted_obb_outputs(x, self.nc, self.n_extra)

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Rearrange raw OBB heads into ``(batch, channels, anchors)`` format.

        Args:
            x: Raw model output tensors.

        Returns:
            Concatenated OBB detection tensor.
        """
        target_count = len(x) // 3
        y_det: list[torch.Tensor] = []
        y_cls: list[torch.Tensor] = []
        y_angle: list[torch.Tensor] = []
        ambiguous: list[tuple[torch.Tensor, list[int]]] = []
        for xi in x:
            if xi.ndim == 3:
                xi = xi.unsqueeze(0)
            elif xi.ndim > 4:
                while xi.ndim > 4 and 1 in xi.shape:
                    xi = xi.squeeze(next(idx for idx, size in enumerate(xi.shape) if size == 1))
                if xi.ndim == 3:
                    xi = xi.unsqueeze(0)
            if xi.ndim != 4:
                raise ValueError(f"Expected 3D or 4D OBB head, got shape {tuple(xi.shape)}.")

            candidates: list[tuple[int, torch.Tensor]] = []
            if xi.shape[1] in {self.reg_max * 4, self.nc, self.n_extra}:
                candidates.append((int(xi.shape[1]), xi))
            if xi.shape[-1] in {self.reg_max * 4, self.nc, self.n_extra}:
                candidates.append((int(xi.shape[-1]), xi.permute(0, 3, 1, 2)))

            deduped: list[tuple[int, torch.Tensor]] = []
            seen_channels: set[int] = set()
            for channel_count, candidate in candidates:
                if channel_count not in seen_channels:
                    seen_channels.add(channel_count)
                    deduped.append((channel_count, candidate))

            if len(candidates) == 2 and len(deduped) == 1:
                channel_count, _ = deduped[0]
                normalized = xi.permute(0, 3, 1, 2)
                if channel_count == self.reg_max * 4:
                    y_det.append(normalized)
                elif channel_count == self.nc:
                    y_cls.append(normalized)
                elif channel_count == self.n_extra:
                    y_angle.append(normalized)
                else:
                    raise ValueError(f"Wrong shape of input: {xi.shape}")
            elif len(deduped) == 1:
                channel_count, normalized = deduped[0]
                if channel_count == self.reg_max * 4:
                    y_det.append(normalized)
                elif channel_count == self.nc:
                    y_cls.append(normalized)
                elif channel_count == self.n_extra:
                    y_angle.append(normalized)
                else:
                    raise ValueError(f"Wrong shape of input: {xi.shape}")
            elif len(deduped) > 1:
                ambiguous.append((xi, [channel_count for channel_count, _ in deduped]))
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        for xi, channel_options in ambiguous:
            if self.reg_max * 4 in channel_options and len(y_det) < target_count:
                y_det.append(xi if xi.shape[1] == self.reg_max * 4 else xi.permute(0, 3, 1, 2))
                continue
            if self.nc in channel_options and len(y_cls) < target_count:
                y_cls.append(xi if xi.shape[1] == self.nc else xi.permute(0, 3, 1, 2))
                continue
            if self.n_extra in channel_options and len(y_angle) < target_count:
                y_angle.append(xi if xi.shape[1] == self.n_extra else xi.permute(0, 3, 1, 2))
                continue
            raise ValueError(f"Wrong shape of input: {xi.shape}")

        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)
        y_angle = sorted(y_angle, key=lambda x: x.numel(), reverse=True)
        if not (len(y_det) == len(y_cls) == len(y_angle)):
            raise ValueError("OBB output arguments are not in a proper form.")
        return torch.cat(
            [
                torch.cat((yi_det, yi_cls, yi_angle), dim=1).flatten(2)
                for yi_det, yi_cls, yi_angle in zip(y_det, y_cls, y_angle)
            ],
            dim=-1,
        )

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode every OBB anchor without confidence filtering."""
        box, scores, angle = torch.split(x, [self.reg_max * 4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        angle = self._angle_from_raw(angle)
        rbox = dist2rbox(self.dfl(box), angle, anchors, dim=1) * stride
        return torch.cat([rbox, scores.sigmoid(), angle], dim=1)

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        """Processes OBB results for a single image.

        Args:
            box_cls: Raw detections for one image.

        Returns:
            Decoded rotated boxes, scores, and angle.
        """
        ic = torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]
        if box_cls.numel() == 0:
            return torch.zeros((4 + self.nc + self.n_extra, 0), dtype=torch.float32, device=self.device)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, angle = torch.split(box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1)
        angle = self._angle_from_raw(angle)
        rbox = dist2rbox(self.dfl(box), angle, anchors[:, ic], dim=1) * stride[:, ic]
        return torch.cat([rbox, scores.sigmoid(), angle], dim=1).squeeze(0)

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters converted OBB outputs from a single ONNX tensor.

        Args:
            x: Converted output tensor.

        Returns:
            Per-image row-major tensors in ``cx, cy, w, h, scores..., angle`` format.
        """
        while x.ndim == 4 and 1 in (x.shape[0], x.shape[1]):
            if x.shape[0] == 1:
                x = x.squeeze(0)
            elif x.shape[1] == 1:
                x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D converted tensor, got shape {tuple(x.shape)}.")
        expected_dim = 4 + self.nc + self.n_extra
        if x.shape[-1] == expected_dim:
            normalized = x
        elif x.shape[1] == expected_dim:
            normalized = x.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported converted tensor shape {tuple(x.shape)}.")

        outputs = []
        for xi in torch.split(normalized, 1, dim=0):
            xi = xi.squeeze(0)
            ic = torch.amax(xi[:, 4 : 4 + self.nc], dim=1) > self.conf_thres
            if torch.any(ic):
                outputs.append(xi[ic])
            else:
                outputs.append(torch.zeros((0, expected_dim), dtype=xi.dtype, device=xi.device))
        return outputs

    def _nms_single(self, xi: torch.Tensor, max_det: int, max_nms: int, max_wh: int) -> torch.Tensor:
        """Apply rotated NMS to a single decoded OBB image tensor."""
        if xi.numel() == 0:
            return torch.zeros((0, 7), dtype=torch.float32, device=self.device)
        xi_t = xi.transpose(0, 1)
        return self._nms_single_legacy_rows(xi_t, max_det=max_det, max_nms=max_nms, max_wh=max_wh)

    def _nms_single_legacy_rows(self, xi: torch.Tensor, max_det: int, max_nms: int, max_wh: int) -> torch.Tensor:
        """Apply rotated NMS to row-major OBB detections."""
        if xi.numel() == 0:
            return torch.zeros((0, 7), dtype=torch.float32, device=self.device)
        xi_out = yolo_multilabel_candidates(xi, self.nc, self.n_extra, self.conf_thres)
        if xi_out.numel() == 0:
            return torch.zeros((0, 7), dtype=torch.float32, device=self.device)
        xi_out = xi_out[torch.argsort(xi_out[:, 4], descending=True)[:max_nms]]
        c = xi_out[:, 5:6] * max_wh
        boxes = torch.cat([xi_out[:, :2] + c, xi_out[:, 2:4], xi_out[:, 6:7]], dim=-1)
        keep = rotated_nms(boxes, xi_out[:, 4], self.iou_thres)[:max_det]
        return xi_out[keep]
