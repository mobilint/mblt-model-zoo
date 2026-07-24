"""Common postprocessing utility functions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeGuard, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..datasets import get_coco_inv, get_dotav1_label

RatioPad: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


def _is_ratio_pad(value: object) -> TypeGuard[RatioPad]:
    """Return whether a value has the nested numeric shape of one RatioPad."""

    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(isinstance(component, (int, float)) for component in pair)
            for pair in value
        )
    )


def normalize_image_shapes(
    image_shapes: tuple[int, int] | Sequence[tuple[int, int]],
    batch_size: int | None = None,
) -> list[tuple[int, int]]:
    """Normalize one or many image shapes to a list, optionally validating its batch size."""

    if len(image_shapes) == 2 and isinstance(image_shapes[0], int):
        shapes = [(int(image_shapes[0]), int(image_shapes[1]))]  # type: ignore[index]
        if batch_size is not None:
            shapes *= batch_size
    else:
        shapes = [(int(shape[0]), int(shape[1])) for shape in image_shapes]  # type: ignore[union-attr]
    if batch_size is not None and len(shapes) != batch_size:
        raise ValueError(f"Expected {batch_size} image shapes, got {len(shapes)}.")
    return shapes


def normalize_ratio_pads(
    ratio_pads: RatioPad | Sequence[RatioPad | None] | None,
    batch_size: int,
) -> list[RatioPad | None]:
    """Normalize optional letterbox metadata to a batch-sized list."""

    if ratio_pads is None:
        return [None] * batch_size
    if _is_ratio_pad(ratio_pads):
        return [ratio_pads] * batch_size
    pads: list[RatioPad | None] = []
    for ratio_pad in ratio_pads:
        if ratio_pad is None:
            pads.append(None)
        elif _is_ratio_pad(ratio_pad):
            pads.append(ratio_pad)
        else:
            raise TypeError("Each ratio_pad must be a ((ratio_x, ratio_y), (pad_x, pad_y)) tuple or None.")
    if len(pads) != batch_size:
        raise ValueError(f"Expected {batch_size} ratio_pad values, got {len(pads)}.")
    return pads


# --- Box Conversion Utilities ---
@overload
def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Converts numpy boxes from ``xywh`` to ``xyxy`` format."""


@overload
def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts torch boxes from ``xywh`` to ``xyxy`` format."""


def xywh2xyxy(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Converts bounding box coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x: Input bounding boxes in (cx, cy, w, h) format.

    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    if isinstance(x, np.ndarray):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    if isinstance(x, torch.Tensor):
        y = torch.clone(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    raise ValueError("x should be np.ndarray or torch.Tensor")


@overload
def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """Converts numpy boxes from ``xyxy`` to ``xywh`` format."""


@overload
def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Converts torch boxes from ``xyxy`` to ``xywh`` format."""


def xyxy2xywh(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Converts bounding box coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    (cx, cy) is the center of the bounding box.

    Args:
        x: Input bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    if isinstance(x, np.ndarray):
        y = np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    if isinstance(x, torch.Tensor):
        y = torch.clone(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    raise ValueError("x should be np.ndarray or torch.Tensor")


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """
    Transform distance (ltrb) to bounding box (xywh or xyxy).
    Args:
        distance (torch.Tensor): Distance from anchor points to box boundaries
            (left, top, right, bottom).
        anchor_points (torch.Tensor): Anchor points (center points).
        xywh (bool, optional): If True, return boxes in (cx, cy, w, h) format.
            If False, return in (x1, y1, x2, y2) format. Defaults to True.
        dim (int, optional): Dimension along which to chunk the distance tensor. Defaults to -1.
    Returns:
        torch.Tensor: Transformed bounding boxes.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)  # xywh bbox
    else:
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def dist2rbox(distance: torch.Tensor, angle: torch.Tensor, anchor_points: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Decode rotated boxes from anchor-relative distances and angles.

    Args:
        distance: Distance tensor in ``ltrb`` format.
        angle: Rotation angle tensor in radians.
        anchor_points: Anchor center points.
        dim: Dimension along which box channels are split.

    Returns:
        Rotated boxes in ``cx, cy, w, h`` format.
    """
    lt, rb = distance.split(2, dim=dim)
    cos_value = torch.cos(angle)
    sin_value = torch.sin(angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x = xf * cos_value - yf * sin_value
    y = xf * sin_value + yf * cos_value
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


@overload
def xywhr2xyxyxyxy(x: np.ndarray) -> np.ndarray:
    """Converts numpy OBBs from ``xywhr`` to polygon corners."""


@overload
def xywhr2xyxyxyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts torch OBBs from ``xywhr`` to polygon corners."""


def xywhr2xyxyxyxy(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Converts oriented boxes from ``cx, cy, w, h, angle`` to four corner points.

    Args:
        x: Oriented boxes with shape ``(..., 5)`` and angle in radians.

    Returns:
        Corner points with shape ``(..., 4, 2)``.
    """
    if isinstance(x, torch.Tensor):
        ctr = x[..., :2]
        w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
        cos_value = torch.cos(angle)
        sin_value = torch.sin(angle)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        return torch.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2], dim=-2)

    if isinstance(x, np.ndarray):
        ctr = x[..., :2]
        w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
        cos_value = np.cos(angle)
        sin_value = np.sin(angle)
        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
        return np.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2], axis=-2)

    raise ValueError("x should be np.ndarray or torch.Tensor")


def xyxyxyxy2xywhr(points: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Converts OBB corner points to regularized ``xywhr`` boxes.

    Args:
        points: Corner points with shape ``(..., 4, 2)``.

    Returns:
        Rotated boxes in ``cx, cy, w, h, angle`` format.
    """
    is_torch = isinstance(points, torch.Tensor)
    points_np = points.detach().cpu().numpy() if is_torch else np.asarray(points)
    flat_points = points_np.reshape(-1, 4, 2).astype(np.float32)
    rboxes = []
    for pts in flat_points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    result_np = np.asarray(rboxes, dtype=points_np.dtype).reshape(*points_np.shape[:-2], 5)
    if is_torch:
        return torch.tensor(result_np, device=points.device, dtype=points.dtype)
    return result_np


def regularize_rboxes(rboxes: torch.Tensor) -> torch.Tensor:
    """Regularize rotated boxes to the angle range ``[0, pi / 2)``.

    Args:
        rboxes: Rotated boxes in ``xywhr`` format.

    Returns:
        Regularized rotated boxes.
    """
    x, y, w, h, angle = rboxes.unbind(dim=-1)
    swap = angle % math.pi >= math.pi / 2
    regularized_w = torch.where(swap, h, w)
    regularized_h = torch.where(swap, w, h)
    regularized_angle = angle % (math.pi / 2)
    return torch.stack([x, y, regularized_w, regularized_h, regularized_angle], dim=-1)


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Gaussian covariance components for probabilistic OBB IoU."""
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos_value = c.cos()
    sin_value = c.sin()
    cos2 = cos_value.pow(2)
    sin2 = sin_value.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos_value * sin_value


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """Calculate pairwise probabilistic IoU for oriented boxes.

    Args:
        obb1: First set of OBBs in ``xywhr`` format with shape ``(N, 5)``.
        obb2: Second set of OBBs in ``xywhr`` format with shape ``(M, 5)``.
        eps: Small value used for numerical stability.

    Returns:
        Pairwise OBB similarities with shape ``(N, M)``.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    obb2 = obb2.to(device=obb1.device, dtype=obb1.dtype)

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denominator) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    iou_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = batch_probiou,
) -> torch.Tensor:
    """Apply fast rotated NMS using an upper-triangular pairwise IoU matrix.

    Args:
        boxes: OBBs in ``xywhr`` format.
        scores: Confidence scores.
        iou_threshold: IoU threshold for suppression.
        iou_func: Pairwise IoU function.

    Returns:
        Kept indices into the original inputs.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[sorted_idx]
    ious = iou_func(sorted_boxes, sorted_boxes).triu_(diagonal=1)
    keep = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
    return sorted_idx[keep]


# --- Detection Utilities ---
def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, max_output: int) -> list[int]:
    """
    Modified non-maximum suppression (NMS) implemented with PyTorch.
    Args:
        boxes (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
        scores (torch.Tensor): Confidence scores for each box (assumed to be sorted in
            descending order).
        iou_threshold (float): IoU threshold for suppression.
        max_output (int): Maximum number of boxes to keep.
    Returns:
        list[int]: Indices of the boxes that have been kept after NMS.
    """
    if boxes.numel() == 0:
        return []
    # Coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    picked_indices: list[int] = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x) * (end_y - start_y)
    # Create an index order (assumed scores are already sorted in descending order)
    order = torch.arange(scores.size(0)).to(boxes.device)
    while order.numel() > 0 and len(picked_indices) < max_output:
        # The index with the highest score
        index = int(order[0].item())
        picked_indices.append(index)
        order = order[1:]  # Remove the index from the order
        if order.numel() == 0 or len(picked_indices) >= max_output:
            break
        # Compute the coordinates of the intersection boxes
        x1 = torch.maximum(start_x[index], start_x[order])
        y1 = torch.maximum(start_y[index], start_y[order])
        x2 = torch.minimum(end_x[index], end_x[order])
        y2 = torch.minimum(end_y[index], end_y[order])
        # Compute width and height of the intersection boxes
        w = torch.clamp(x2 - x1, min=0.0)
        h = torch.clamp(y2 - y1, min=0.0)
        intersection = w * h
        # Compute the IoU ratio
        union = areas[index] + areas[order] - intersection
        ratio = intersection / union
        # Keep boxes with IoU less than or equal to the threshold
        keep = (ratio <= iou_threshold).to(order.device)
        order = order[keep]
    return picked_indices


def dual_topk(
    pre_topk: torch.Tensor,
    nc: int,
    n_extra: int,
    max_det: int = 300,
    conf_thres: float = 0.25,
    score_is_logits: bool = False,
) -> torch.Tensor:
    """
    Perform dual-stage topk selection for NMS-free models.
    Args:
        pre_topk (torch.Tensor): Input tensor of shape (*, 4 + nc + n_extra).
        nc (int): Number of classes.
        n_extra (int): Number of extra elements (e.g., masks, keypoints).
        max_det (int): Maximum detections to keep. Defaults to 300.
        conf_thres (float): Confidence threshold. Defaults to 0.25.
        score_is_logits (bool): Whether class scores are logits. When true, apply
            the confidence cutoff and both rankings before sigmoid, then convert
            only selected scores to probabilities. Defaults to false.
    Returns:
        torch.Tensor: Filtered detections of shape (*, 6 + n_extra).
    """
    score_start = 4
    score_end = 4 + nc
    score_view = pre_topk[:, score_start:score_end]
    threshold = math.log(conf_thres / (1.0 - conf_thres)) if score_is_logits else conf_thres
    ic = score_view.amax(dim=-1) > threshold
    pre_topk = pre_topk[ic]

    if pre_topk.shape[0] == 0:
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device)
    max_det = min(pre_topk.shape[0], max_det)

    row_index = torch.topk(pre_topk[:, score_start:score_end].amax(dim=-1), max_det, dim=0).indices
    selected = pre_topk[row_index]
    top_scores, flat_index = torch.topk(selected[:, score_start:score_end].reshape(-1), max_det)
    keep = top_scores > threshold
    if not torch.any(keep):
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device)

    top_scores = top_scores[keep]
    flat_index = flat_index[keep]
    box_index = flat_index // nc
    labels = (flat_index % nc).to(selected.dtype).unsqueeze(-1)

    output = torch.empty((top_scores.shape[0], 6 + n_extra), dtype=selected.dtype, device=selected.device)
    output[:, :4] = selected[box_index, :4]
    output[:, 4] = top_scores.sigmoid() if score_is_logits else top_scores
    output[:, 5:6] = labels
    if n_extra > 0:
        output[:, 6:] = selected[box_index, score_end:]
    return output


def yolo_multilabel_candidates(
    detections: torch.Tensor,
    nc: int,
    n_extra: int,
    conf_thres: float,
) -> torch.Tensor:
    """Expand YOLO rows into one detection per class score above threshold.

    Args:
        detections: Row-major detections with columns ``box, class scores, extra``.
        nc: Number of classes.
        n_extra: Number of extra channels after class scores.
        conf_thres: Confidence threshold.

    Returns:
        Canonical detection rows with columns ``box, score, class, extra``.
    """
    if detections.numel() == 0:
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=detections.device)

    boxes = detections[:, :4]
    scores = detections[:, 4 : 4 + nc]
    extra = detections[:, 4 + nc :]
    box_index, class_index = torch.where(scores > conf_thres)
    if box_index.numel() == 0:
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=detections.device)

    output = torch.empty((box_index.numel(), 6 + n_extra), dtype=detections.dtype, device=detections.device)
    output[:, :4] = boxes[box_index]
    output[:, 4] = scores[box_index, class_index]
    output[:, 5] = class_index.to(detections.dtype)
    if n_extra > 0:
        output[:, 6:] = extra[box_index]
    return output


def normalize_converted_obb_part(x: torch.Tensor, channel_count: int) -> torch.Tensor:
    """Normalize a converted OBB output part to ``(batch, anchors, channels)``.

    Args:
        x: Converted output part from a model runtime.
        channel_count: Expected feature-channel count for this part.

    Returns:
        The normalized row-major tensor.
    """
    while x.ndim > 3:
        singleton_dims = [idx for idx, size in enumerate(x.shape) if idx != 0 and size == 1]
        if not singleton_dims:
            raise ValueError(f"Expected converted OBB part with up to 3 non-batch dimensions, got {tuple(x.shape)}.")
        x = x.squeeze(singleton_dims[0])
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D converted OBB part, got shape {tuple(x.shape)}.")
    if x.shape[-1] == channel_count:
        return x
    if x.shape[1] == channel_count:
        return x.transpose(1, 2)
    raise ValueError(f"Could not find channel count {channel_count} in converted OBB part with shape {tuple(x.shape)}.")


def concat_converted_obb_outputs(x: list[torch.Tensor], nc: int, n_extra: int) -> torch.Tensor:
    """Concatenate converted OBB box, class, and angle outputs in canonical order.

    Args:
        x: Converted OBB runtime outputs.
        nc: Number of OBB classes.
        n_extra: Number of extra OBB channels.

    Returns:
        Detections in ``cx, cy, w, h, class scores..., angle`` format.
    """
    if len(x) == 1:
        return x[0]
    if len(x) != 3:
        raise ValueError(f"Expected 1 or 3 converted OBB outputs, got {len(x)}.")

    expected_parts = {"box": 4, "scores": nc, "angle": n_extra}
    parts: dict[str, torch.Tensor] = {}
    for xi in x:
        matches: list[tuple[str, torch.Tensor]] = []
        for name, channel_count in expected_parts.items():
            try:
                matches.append((name, normalize_converted_obb_part(xi, channel_count)))
            except ValueError:
                continue
        if len(matches) != 1:
            match_names = ", ".join(name for name, _ in matches) or "none"
            raise ValueError(
                f"Could not uniquely classify converted OBB output {tuple(xi.shape)}; matches: {match_names}."
            )
        name, normalized = matches[0]
        if name in parts:
            raise ValueError(f"Duplicate converted OBB {name} output.")
        parts[name] = normalized

    missing = [name for name in expected_parts if name not in parts]
    if missing:
        raise ValueError(f"Missing converted OBB outputs: {', '.join(missing)}.")
    return torch.cat([parts["box"], parts["scores"], parts["angle"]], dim=-1)


def decode_split_converted_obb_outputs(
    x: list[torch.Tensor],
    nc: int,
    n_extra: int,
    anchors: torch.Tensor,
    stride: torch.Tensor,
) -> torch.Tensor:
    """Decode MXQ decode-true OBB outputs split into score, angle, and coordinate tensors.

    Args:
        x: Five converted runtime outputs: class scores, rotation angle, and
            coordinate tensors containing decoded ``wh`` and pre-rotated center offsets.
        nc: Number of OBB classes.
        n_extra: Number of extra OBB channels.
        anchors: Anchor points in ``(2, anchors)`` format.
        stride: Stride tensor in ``(1, anchors)`` format.

    Returns:
        Detections in ``cx, cy, w, h, class scores..., angle`` format.
    """
    if n_extra != 1:
        raise ValueError(f"Expected one OBB angle channel, got n_extra={n_extra}.")
    if len(x) != 5:
        raise ValueError(f"Expected five split converted OBB outputs, got {len(x)}.")

    try:
        scores = normalize_converted_obb_part(x[0], nc)
        angle = normalize_converted_obb_part(x[1], n_extra)
    except ValueError:
        scores = normalize_converted_obb_part(x[1], nc)
        angle = normalize_converted_obb_part(x[0], n_extra)
    wh = normalize_converted_obb_part(x[2], 2)
    x_offset = normalize_converted_obb_part(x[3], 1)
    y_offset = normalize_converted_obb_part(x[4], 1)
    cos_value = torch.cos(angle)
    sin_value = torch.sin(angle)
    center_offset = torch.cat(
        [
            x_offset * cos_value - y_offset * sin_value,
            x_offset * sin_value + y_offset * cos_value,
        ],
        dim=-1,
    )
    anchors_t = anchors.transpose(0, 1).unsqueeze(0).to(device=wh.device, dtype=wh.dtype)
    stride_t = stride.transpose(0, 1).unsqueeze(0).to(device=wh.device, dtype=wh.dtype)
    if anchors_t.shape[1] < wh.shape[1]:
        raise ValueError(f"Got {wh.shape[1]} OBB coordinate rows but only {anchors_t.shape[1]} anchors.")
    anchors_t = anchors_t[:, : wh.shape[1]]
    stride_t = stride_t[:, : wh.shape[1]]
    box = torch.cat([anchors_t + center_offset, wh], dim=-1) * stride_t
    return torch.cat([box, scores, angle], dim=-1)


# --- Scaling & Clipping Utilities ---
@overload
def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> np.ndarray: ...


@overload
def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> torch.Tensor: ...


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: np.ndarray | torch.Tensor,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Original Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L92
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they
    were originally specified in (img1_shape) to the shape of a different image (img0_shape).
    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for,
            in the format of (height, width).
        boxes (np.ndarray | torch.Tensor): the bounding boxes of the objects in the image,
            in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes.
            If not provided, the ratio and pad will be calculated based on the size
            difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by
            yolo style. If False then do regular rescaling.
    Returns:
        np.ndarray | torch.Tensor: The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)
    if isinstance(boxes, np.ndarray):
        if padding:
            boxes[..., [0, 2]] -= pad[0]  # x padding
            boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return clip_boxes(boxes, img0_shape)
    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


@overload
def scale_coords(
    img1_shape: tuple[int, int],
    coords: np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> np.ndarray: ...


@overload
def scale_coords(
    img1_shape: tuple[int, int],
    coords: torch.Tensor,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> torch.Tensor: ...


def scale_coords(
    img1_shape: tuple[int, int],
    coords: np.ndarray | torch.Tensor,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Original Source:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L756
    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        coords (np.ndarray | torch.Tensor): The coordinates of the objects in the image, in the format of (x, y).
        img0_shape (tuple): The shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    Returns:
        np.ndarray | torch.Tensor: The scaled coordinates, in the format of (x, y)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            (img1_shape[1] - round(img0_shape[1] * gain)) / 2,
            (img1_shape[0] - round(img0_shape[0] * gain)) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if isinstance(coords, np.ndarray):
        if padding:
            coords[..., 0] -= pad[0]  # x padding
            coords[..., 1] -= pad[1]  # y padding
        coords[..., :2] /= gain
        return clip_coords(coords, img0_shape)
    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., :2] /= gain
    return clip_coords(coords, img0_shape)


def scale_rboxes(
    img1_shape: tuple[int, int],
    rboxes: torch.Tensor,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescale rotated boxes from model input size to an original image size.

    Args:
        img1_shape: Processed image shape.
        rboxes: Rotated boxes in ``xywhr`` format.
        img0_shape: Original image shape.
        ratio_pad: Optional precomputed resize ratio and padding.
        padding: Whether YOLO-style letterbox padding was applied.

    Returns:
        Rescaled rotated boxes in ``xywhr`` format.
    """
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)
    scaled = rboxes.clone()
    if padding:
        scaled[..., 0] -= pad[0]
        scaled[..., 1] -= pad[1]
    scaled[..., :4] /= gain
    return scaled


def compute_ratio_pad(
    img1_shape: tuple[int, int],
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[float, tuple[float, float]]:
    """Computes ratio and padding used to resize an image to the input shape.

    Args:
        img1_shape (tuple): The target shape (height, width).
        img0_shape (tuple): The original shape (height, width).
        ratio_pad (tuple, optional): Pre-calculated (ratio, pad) tuple.
            If None, it will be calculated from the shapes. Defaults to None.

    Returns:
        tuple: (gain, pad) where gain is the scaling factor and pad is the (x, y) padding.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1),
            round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad


@overload
def clip_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray: ...


@overload
def clip_boxes(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor: ...


def clip_boxes(boxes: np.ndarray | torch.Tensor, shape: tuple[int, int]) -> np.ndarray | torch.Tensor:
    """
    Clip bounding boxes to image shape.
    Args:
        boxes (np.ndarray | torch.Tensor): Bounding boxes.
        shape (tuple): Image shape (height, width).
    Returns:
        np.ndarray | torch.Tensor: Clipped bounding boxes.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    else:
        boxes[..., 0] = np.clip(boxes[..., 0], 0, shape[1])
        boxes[..., 1] = np.clip(boxes[..., 1], 0, shape[0])
        boxes[..., 2] = np.clip(boxes[..., 2], 0, shape[1])
        boxes[..., 3] = np.clip(boxes[..., 3], 0, shape[0])
    return boxes


@overload
def clip_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray: ...


@overload
def clip_coords(coords: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor: ...


def clip_coords(coords: np.ndarray | torch.Tensor, shape: tuple[int, int]) -> np.ndarray | torch.Tensor:
    """Clips coordinates to the image shape.

    Args:
        coords (np.ndarray | torch.Tensor): Coordinates to clip.
        shape (tuple): Image shape (height, width).

    Returns:
        np.ndarray | torch.Tensor: Clipped coordinates.
    """
    if isinstance(coords, torch.Tensor):
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])
    else:
        coords[..., 0] = np.clip(coords[..., 0], 0, shape[1])
        coords[..., 1] = np.clip(coords[..., 1], 0, shape[0])
    return coords


# --- Segmentation Utilities ---
def process_mask(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
    bboxes: torch.Tensor,
    shape: tuple[int, int],
    upsample: bool = False,
) -> torch.Tensor:
    """Processes masks by applying coefficients to prototypes and cropping.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L680

    Args:
        protos (torch.Tensor): Prototype masks of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): Mask coefficients of shape [n, mask_dim].
        bboxes (torch.Tensor): Bounding boxes of shape [n, 4].
        shape (tuple): Input image size (h, w).
        upsample (bool, optional): Whether to upsample the masks to the original image size.
            Defaults to False.

    Returns:
        torch.Tensor: Processed binary masks.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # n, CHW
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_upsample(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
    bboxes: torch.Tensor,
    shape: tuple[int, int] | list[int],
) -> torch.Tensor:
    """Applies masks to bounding boxes with upsampling for higher quality.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L713
    This produces higher quality masks than `process_mask` but is slower.

    Args:
        protos (torch.Tensor): Prototype masks of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): Mask coefficients of shape [n, mask_dim].
        bboxes (torch.Tensor): Bounding boxes of shape [n, 4].
        shape (tuple): Target image size (h, w).

    Returns:
        torch.Tensor: Upsampled and thresholded binary masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # n, CHW
    masks = scale_masks(masks, (shape[0], shape[1]))  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Crops masks to bounding boxes.

    Args:
        masks (torch.Tensor): Masks of shape [n, h, w].
        boxes (torch.Tensor): Bounding boxes of shape [n, 4] in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: Cropped masks.
    """
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((rows >= x1) * (rows < x2) * (cols >= y1) * (cols < y2))


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescales segment masks to the target shape.

    Args:
        masks (torch.Tensor): Input masks of shape (C, H, W).
        shape (tuple): Target shape (height, width).
        ratio_pad (tuple, optional): Pre-calculated (ratio, pad) tuple.
            If None, it will be calculated from the shapes. Defaults to None.
        padding (bool, optional): If True, assumes the masks were generated from
            an image with YOLO-style padding. Defaults to True.

    Returns:
        torch.Tensor: Rescaled masks of shape (C, target_h, target_w).
    """
    im1_h, im1_w = masks.shape[1:]
    im0_h, im0_w = shape[:2]
    if masks.numel() == 0:
        return torch.zeros((0, im0_h, im0_w), dtype=masks.dtype, device=masks.device)
    if im1_h == im0_h and im1_w == im0_w:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_h / im0_h, im1_w / im0_w)  # gain  = old / new
        pad_w, pad_h = (im1_w - round(im0_w * gain)), (im1_h - round(im0_h * gain))  # wh padding
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom, right = im1_h - round(pad_h + 0.1), im1_w - round(pad_w + 0.1)
    masks = masks[..., top:bottom, left:right]
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)  # 1NHW
    return masks[0]


def to_string(counts: list[int]) -> str:
    """Converts the RLE object into a compact string representation.

    Each count is delta-encoded and variable-length encoded as a string.

    Args:
        counts (list[int]): List of RLE counts.

    Returns:
        str: Compact string representation of the RLE object.
    """
    result = []

    for i, x in enumerate(counts):
        x = int(x)

        # Apply delta encoding for all counts after the second entry
        if i > 2:
            x -= int(counts[i - 2])

        # Variable-length encode the value
        while True:
            c = x & 0x1F  # Take 5 bits
            x >>= 5

            # If the sign bit (0x10) is set, continue if x != -1;
            # otherwise, continue if x != 0
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20  # Set continuation bit
            c += 48  # Shift to ASCII
            result.append(chr(c))
            if not more:
                break

    return "".join(result)


def multi_encode(pixels: torch.Tensor) -> list[list[int]]:
    """Convert multiple binary masks using Run-Length Encoding (RLE).

    Args:
        pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask
            with shape [N, H*W].

    Returns:
        list[list[int]]: A list of RLE counts for each mask.
    """
    pixel_rows = pixels.detach().cpu().numpy().astype(np.uint8, copy=False)
    width = pixel_rows.shape[1]
    counts = []
    for i in range(pixel_rows.shape[0]):
        pixel_row = pixel_rows[i]
        positions = np.flatnonzero(pixel_row[1:] != pixel_row[:-1]) + 1
        if positions.size:
            count = np.diff(positions).tolist()
            count.insert(0, int(positions[0]))
            count.append(int(width - positions[-1]))
        else:
            count = [width]
        if pixel_row[0] == 1:
            count = [0, *count]
        counts.append(count)

    return counts


def nmsout2eval(
    nms_outs: list[torch.Tensor] | torch.Tensor,
    img1_shape: tuple[int, int],
    img0_shapes: tuple[int, int] | Sequence[tuple[int, int]],
    ratio_pads: RatioPad | Sequence[RatioPad | None] | None = None,
) -> tuple[list[list[int]], list[list[list[float]]], list[list[float]]]:
    """Converts NMS output to COCO evaluation format.

    Args:
        nms_outs (list[torch.Tensor] | torch.Tensor): The output of the NMS
            operation of shape (n, 6), where n is the number of objects.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (list[tuple]): Original image shapes [(H, W), ...].

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
    """

    if not isinstance(nms_outs, list):
        nms_outs = [nms_outs]
    actual_img0_shapes = normalize_image_shapes(img0_shapes, len(nms_outs))
    actual_ratio_pads = normalize_ratio_pads(ratio_pads, len(nms_outs))
    labels_list: list[list[int]] = []
    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    for nms_out, img0_shape, ratio_pad in zip(nms_outs, actual_img0_shapes, actual_ratio_pads):
        boxes = nms_out[:, :4].clone()
        scores = nms_out[:, 4]
        labels = nms_out[:, 5]
        boxes = scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=ratio_pad)  # scale boxes to original image size
        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]  # xyxy to xywh with corner xy

        boxes_tolist = [[round(float(value), 3) for value in box] for box in boxes.tolist()]
        scores_tolist = [round(float(score), 5) for score in scores.tolist()]
        labels_tolist = labels.tolist()
        labels_res = [get_coco_inv(int(label)) for label in labels_tolist]

        labels_list.append(labels_res)
        boxes_list.append(boxes_tolist)
        scores_list.append(scores_tolist)

    return labels_list, boxes_list, scores_list


def nmsout2eval_seg(
    nms_outs: Any,
    img1_shape: tuple[int, int],
    img0_shapes: tuple[int, int] | list[tuple[int, int]],
    ratio_pads: RatioPad | list[RatioPad | None] | None = None,
) -> tuple[list[list[int]], list[list[list[float]]], list[list[float]], list[list[dict[str, Any]]]]:
    """Converts segmentation NMS output to COCO evaluation format.

    Args:
        nms_outs (Union[list, tuple]): Segmentation postprocess output in one of two forms:
            `(det_result, seg_result)` for a single image or a list of those pairs for a batch.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (tuple | list[tuple]): Original image shape for a single image or
            a list of original shapes for a batch.

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
            - extra (list[list]): The encoded segmentation masks for each image.
    """
    actual_img0_shapes = normalize_image_shapes(img0_shapes)
    actual_ratio_pads = normalize_ratio_pads(ratio_pads, len(actual_img0_shapes))

    if not isinstance(nms_outs[0], (list, tuple)):
        actual_nms_outs = [nms_outs]
    else:
        actual_nms_outs = nms_outs

    det_results = []
    seg_results = []
    for nms_out in actual_nms_outs:
        det_results.append(nms_out[0])
        seg_results.append(nms_out[1])

    labels_list, boxes_list, scores_list = nmsout2eval(
        det_results,
        img1_shape,
        actual_img0_shapes,
        ratio_pads=actual_ratio_pads,
    )

    scaled_seg_results = [
        scale_masks(seg_result.to(torch.float32), (img0_shape[0], img0_shape[1]), ratio_pad=ratio_pad)
        for seg_result, img0_shape, ratio_pad in zip(seg_results, actual_img0_shapes, actual_ratio_pads)
    ]

    def mask_encode(seg_result: torch.Tensor) -> list[dict[str, Any]]:
        extra = []
        h, w = seg_result.shape[1:3]
        seg_result = seg_result.permute(0, 2, 1).contiguous().view(seg_result.shape[0], h * w).byte()
        counts = multi_encode(seg_result)
        assert len(counts) == seg_result.shape[0], "The number of encoded masks must match the mask tensor batch size."
        for c in counts:
            extra.append({"size": [h, w], "counts": to_string(c)})
        return extra

    extra_list = [mask_encode(seg_result) for seg_result in scaled_seg_results]
    for labels, boxes, scores, extra in zip(labels_list, boxes_list, scores_list, extra_list):
        assert len(labels) == len(boxes) == len(scores) == len(extra), (
            "Segmentation evaluation outputs must have matching detection and mask counts."
        )
    return labels_list, boxes_list, scores_list, extra_list


def nmsout2eval_pose(
    nms_outs: list[torch.Tensor] | torch.Tensor,
    img1_shape: tuple[int, int],
    img0_shapes: tuple[int, int] | list[tuple[int, int]],
    ratio_pads: RatioPad | list[RatioPad | None] | None = None,
) -> tuple[list[list[int]], list[list[list[float]]], list[list[float]], list[list[list[float]]]]:
    """Converts pose estimation NMS output to COCO evaluation format.

    Args:
        nms_outs (list): The output of the NMS operation.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (list[tuple]): Original image shapes [(H, W), ...].

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
            - keypoints (list[list]): The scaled keypoints for each image.
    """
    actual_img0_shapes = normalize_image_shapes(img0_shapes)
    actual_ratio_pads = normalize_ratio_pads(ratio_pads, len(actual_img0_shapes))
    if not isinstance(nms_outs, list):
        actual_nms_outs = [nms_outs]
    else:
        actual_nms_outs = nms_outs
    labels_list, boxes_list, scores_list = nmsout2eval(
        actual_nms_outs,
        img1_shape,
        actual_img0_shapes,
        ratio_pads=actual_ratio_pads,
    )
    extra = [
        scale_coords(img1_shape, nms_out[:, 6:].reshape(-1, 17, 3), img0_shape, ratio_pad=ratio_pad).reshape(-1, 51)
        for nms_out, img0_shape, ratio_pad in zip(actual_nms_outs, actual_img0_shapes, actual_ratio_pads)
    ]
    return labels_list, boxes_list, scores_list, [x.tolist() for x in extra]


def nmsout2eval_obb(
    nms_outs: list[torch.Tensor] | torch.Tensor,
    img1_shape: tuple[int, int],
    img0_shapes: tuple[int, int] | list[tuple[int, int]],
    ratio_pads: RatioPad | list[RatioPad | None] | None = None,
    include_xywhr: bool = False,
) -> tuple[Any, ...]:
    """Converts OBB NMS output to DOTAv1 evaluation format.

    Args:
        nms_outs: Detections with rows ``cx, cy, w, h, score, cls, angle``.
        img1_shape: Processed image shape.
        img0_shapes: Original image shape or shapes.
        ratio_pads: Optional letterbox metadata.
        include_xywhr: Whether to include scaled ``xywhr`` boxes in the return value.

    Returns:
        DOTAv1 labels, polygons, scores, and optionally scaled ``xywhr`` boxes.
    """
    actual_img0_shapes = normalize_image_shapes(img0_shapes)
    actual_ratio_pads = normalize_ratio_pads(ratio_pads, len(actual_img0_shapes))
    actual_nms_outs = [nms_outs] if not isinstance(nms_outs, list) else nms_outs

    labels_list: list[list[str]] = []
    polygons_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    xywhr_list: list[list[list[float]]] = []
    for nms_out, img0_shape, ratio_pad in zip(actual_nms_outs, actual_img0_shapes, actual_ratio_pads):
        if nms_out.numel() == 0:
            labels_list.append([])
            polygons_list.append([])
            scores_list.append([])
            xywhr_list.append([])
            continue

        rboxes = torch.cat([nms_out[:, :4], nms_out[:, 6:7]], dim=-1)
        rboxes = scale_rboxes(img1_shape, rboxes, img0_shape, ratio_pad=ratio_pad)
        polygons = xywhr2xyxyxyxy(rboxes).reshape(-1, 8)
        polygons = scale_coords(img0_shape, polygons.reshape(-1, 4, 2), img0_shape).reshape(-1, 8)

        labels = [get_dotav1_label(int(label)) for label in nms_out[:, 5].tolist()]
        scores = [round(float(score), 5) for score in nms_out[:, 4].tolist()]
        polygons_tolist = [[round(float(value), 3) for value in polygon] for polygon in polygons.tolist()]
        xywhr_tolist = [[round(float(value), 3) for value in rbox] for rbox in rboxes.tolist()]

        labels_list.append(labels)
        polygons_list.append(polygons_tolist)
        scores_list.append(scores)
        xywhr_list.append(xywhr_tolist)

    if include_xywhr:
        return labels_list, polygons_list, scores_list, xywhr_list
    return labels_list, polygons_list, scores_list


class YOLOSegPostMixin:
    """Mixin class for YOLO segmentation postprocessing."""

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
        ratio_pad: RatioPad | list[RatioPad | None] | None = None,
    ) -> tuple[Any, ...]:
        """Converts NMS output to evaluation format for segmentation.

        Args:
            nms_out: NMS output (detections and prototypes).
            img1_shape: Resized image shape.
            img0_shape: List of original image shapes.

        Returns:
            Tuple: (labels_list, boxes_list, scores_list, extra_list).
        """
        return nmsout2eval_seg(nms_out, img1_shape, img0_shape, ratio_pads=ratio_pad)


class YOLOPosePostMixin:
    """Mixin class for YOLO pose estimation postprocessing."""

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
        ratio_pad: RatioPad | list[RatioPad | None] | None = None,
    ) -> tuple[Any, ...]:
        """Converts NMS output to evaluation format for pose estimation.

        Args:
            nms_out: NMS output (detections with keypoints).
            img1_shape: Resized image shape.
            img0_shape: List of original image shapes.

        Returns:
            Tuple: (labels_list, boxes_list, scores_list, extra_list).
        """
        return nmsout2eval_pose(nms_out, img1_shape, img0_shape, ratio_pads=ratio_pad)


class YOLOOBBPostMixin:
    """Mixin class for YOLO oriented-bounding-box postprocessing."""

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
        ratio_pad: RatioPad | list[RatioPad | None] | None = None,
        include_xywhr: bool = False,
    ) -> tuple[Any, ...]:
        """Converts OBB detections to DOTAv1 labels, polygons, and scores.

        Args:
            nms_out: NMS output with rows ``cx, cy, w, h, score, cls, angle``.
            img1_shape: Resized image shape.
            img0_shape: Original image shape or shapes.
            ratio_pad: Optional letterbox metadata.
            include_xywhr: Whether to include scaled rotated boxes.

        Returns:
            DOTAv1 labels, polygons, scores, and optionally scaled ``xywhr`` boxes.
        """
        return nmsout2eval_obb(nms_out, img1_shape, img0_shape, ratio_pads=ratio_pad, include_xywhr=include_xywhr)
