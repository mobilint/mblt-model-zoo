from __future__ import annotations

from typing import Any, cast

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
    dual_topk,
    rotated_nms,
    yolo_multilabel_candidates,
)


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

    def _normalize_converted_part(self, x: torch.Tensor, channel_count: int) -> torch.Tensor | None:
        """Normalize a split decode-true part to ``(B, anchors, channels)``."""

        while x.ndim > 3 and 1 in x.shape:
            x = x.squeeze(next(idx for idx, size in enumerate(x.shape) if size == 1))

        if x.ndim == 2:
            if x.shape[-1] == channel_count:
                return x.unsqueeze(0)
            if x.shape[0] == channel_count:
                return x.transpose(0, 1).unsqueeze(0)
            return None

        if x.ndim == 3 and x.shape[0] == 1:
            if x.shape[-1] == channel_count:
                return x
            if x.shape[1] == channel_count:
                return x.transpose(1, 2)

        return None

    def _collect_converted_parts(
        self,
        x: list[torch.Tensor],
        *,
        require_extra: bool,
    ) -> tuple[torch.Tensor, set[int]] | None:
        """Collect decode-true box/class/extra parts while ignoring reducemax."""

        part_by_channels: dict[int, torch.Tensor] = {}
        used_indices: set[int] = set()
        required_channels = [4, self.nc]
        if require_extra:
            required_channels.append(self.n_extra)

        for idx, xi in enumerate(x):
            for channel_count in required_channels:
                if channel_count in part_by_channels:
                    continue
                normalized = self._normalize_converted_part(xi, channel_count)
                if normalized is None:
                    continue
                part_by_channels[channel_count] = normalized
                used_indices.add(idx)
                break

        if any(channel_count not in part_by_channels for channel_count in required_channels):
            return None

        batch_size = part_by_channels[4].shape[0]
        anchor_count = part_by_channels[4].shape[1]
        for channel_count in required_channels[1:]:
            part = part_by_channels[channel_count]
            if part.shape[0] != batch_size or part.shape[1] != anchor_count:
                return None

        ordered_parts = [part_by_channels[4], part_by_channels[self.nc]]
        if require_extra:
            ordered_parts.append(part_by_channels[self.n_extra])
        return torch.cat(ordered_parts, dim=-1), used_indices

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
        if not outputs:
            raise ValueError("At least one output tensor is required.")

        output_dim = int(outputs[0].shape[1])
        padded_outputs = []
        for output in outputs:
            if output.ndim != 2:
                raise ValueError(f"Expected 2D detection rows, got shape {tuple(output.shape)}.")
            if output.shape[1] != output_dim:
                raise ValueError(f"Inconsistent detection row width {output.shape[1]}; expected {output_dim}.")
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
        if len(x) in {2, 3}:
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
        converted_parts = self._collect_converted_parts(x, require_extra=self.n_extra > 0)
        if converted_parts is not None:
            converted, _ = converted_parts
            return converted

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
        if len(x) in {4, 5}:
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

        converted_parts = self._collect_converted_parts(x, require_extra=True)
        if converted_parts is not None:
            converted, used_indices = converted_parts
            proto_candidates = []
            for idx, xi in enumerate(x):
                if idx in used_indices:
                    continue
                proto = xi
                if proto.ndim == 3:
                    proto = proto.unsqueeze(0)
                if proto.ndim == 4 and (proto.shape[-1] == self.n_extra or proto.shape[1] == self.n_extra):
                    proto_candidates.append(proto)
            if len(proto_candidates) == 1:
                return converted, proto_candidates[0]

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
        if len(x) in {3, 4}:
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
        converted_parts = self._collect_converted_parts(x, require_extra=True)
        if converted_parts is not None:
            converted, _ = converted_parts
            return converted

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


class YOLODFLFreeOBBPost(YOLOOBBPostMixin, YOLODFLFreePost):
    """Postprocessing for DFL-free YOLO OBB models."""

    def _pre_process(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Preprocess OBB inputs into row-major detections.

        Args:
            x: Raw model outputs.

        Returns:
            A tuple of detections and no prototype output.
        """
        if len(x) in {1, 3, 5}:
            converted = cast(torch.Tensor, self.conversion(x))
            return self.filter_conversion(converted), None
        rearranged = self.rearrange(x)
        if not isinstance(rearranged, torch.Tensor):
            raise TypeError("rearrange should return a tensor for DFL-free OBB postprocessing.")
        return self.decode(rearranged), None

    def conversion(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert DFL-free OBB outputs to a single tensor.

        Args:
            x: Input tensors.

        Returns:
            Converted tensor with last dimension ``4 + nc + 1``.
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
        """Rearrange split raw DFL-free OBB heads.

        Args:
            x: Raw model output tensors.

        Returns:
            Concatenated tensor in ``(batch, channels, anchors)`` format.
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
            if xi.shape[1] in {4, self.nc, self.n_extra}:
                candidates.append((int(xi.shape[1]), xi))
            if xi.shape[-1] in {4, self.nc, self.n_extra}:
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
                if channel_count == 4:
                    y_det.append(normalized)
                elif channel_count == self.nc:
                    y_cls.append(normalized)
                elif channel_count == self.n_extra:
                    y_angle.append(normalized)
                else:
                    raise ValueError(f"Wrong shape of input: {xi.shape}")
            elif len(deduped) == 1:
                channel_count, normalized = deduped[0]
                if channel_count == 4:
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
            if 4 in channel_options and len(y_det) < target_count:
                y_det.append(xi if xi.shape[1] == 4 else xi.permute(0, 3, 1, 2))
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
        """Decode every OBB anchor for export-style output."""
        box, scores, angle = torch.split(x, [4, self.nc, self.n_extra], dim=1)
        anchors = self.anchors_as_tensor().unsqueeze(0)
        stride = self.stride_as_tensor().unsqueeze(0)
        rbox = dist2rbox(box, angle, anchors, dim=1) * stride
        return torch.cat([rbox, scores.sigmoid(), angle], dim=1).transpose(1, 2)

    def process_box_cls(self, box_cls: torch.Tensor) -> torch.Tensor:
        """Processes raw DFL-free OBB results for one image.

        Args:
            box_cls: Raw detections for one image.

        Returns:
            Raw OBB rows ``cx, cy, w, h, class scores, angle`` before NMS.
        """
        ic = torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0) > self.inv_conf_thres
        box_cls = box_cls[:, ic]
        if box_cls.numel() == 0:
            return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32, device=self.device)
        anchors = self.anchors_as_tensor()
        stride = self.stride_as_tensor()
        box, scores, angle = torch.split(box_cls[None], [4, self.nc, self.n_extra], dim=1)
        rbox = dist2rbox(box, angle, anchors[:, ic], dim=1) * stride[:, ic]
        return torch.cat([rbox, scores.sigmoid(), angle], dim=1).squeeze(0).transpose(0, 1)

    def filter_conversion(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Filters converted DFL-free OBB outputs.

        Args:
            x: Converted output tensor.

        Returns:
            Per-image canonical OBB detection rows before rotated NMS.
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
        for xi in normalized:
            keep = xi[:, 4 : 4 + self.nc].amax(dim=1) > self.conf_thres
            if torch.any(keep):
                outputs.append(xi[keep])
            else:
                outputs.append(torch.zeros((0, expected_dim), dtype=xi.dtype, device=xi.device))
        return outputs

    def nms(
        self,
        x: torch.Tensor | list[torch.Tensor],
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ) -> list[torch.Tensor]:
        """Apply rotated NMS to DFL-free OBB detections.

        Args:
            x: Decoded detections.
            max_det: Maximum detections to keep.
            max_nms: Maximum candidates to consider.
            max_wh: Class offset size.

        Returns:
            Per-image OBB detections after rotated NMS.
        """
        detections = x if isinstance(x, list) else list(x)
        output = []
        for xi in detections:
            if xi.numel() == 0:
                output.append(torch.zeros((0, 7), dtype=torch.float32, device=self.device))
                continue
            if xi.shape[1] == 4 + self.nc + self.n_extra:
                xi = yolo_multilabel_candidates(xi, self.nc, self.n_extra, self.conf_thres)
            elif xi.shape[1] == 6 + self.n_extra:
                xi = xi[xi[:, 4] > self.conf_thres]
            else:
                raise ValueError(f"Unsupported OBB detection shape {tuple(xi.shape)}.")
            if xi.numel() == 0:
                output.append(torch.zeros((0, 7), dtype=torch.float32, device=self.device))
                continue
            xi = xi[torch.argsort(xi[:, 4], descending=True)[:max_nms]]
            c = xi[:, 5:6] * max_wh
            boxes = torch.cat([xi[:, :2] + c, xi[:, 2:4], xi[:, 6:7]], dim=-1)
            keep = rotated_nms(boxes, xi[:, 4], self.iou_thres)[:max_det]
            output.append(xi[keep])
        return output
