"""Common postprocessing utility functions."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import torch
import torch.nn.functional as F

from ..datasets import get_coco_inv


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
) -> torch.Tensor:
    """
    Perform dual-stage topk selection for NMS-free models.
    Args:
        pre_topk (torch.Tensor): Input tensor of shape (*, 4 + nc + n_extra).
        nc (int): Number of classes.
        n_extra (int): Number of extra elements (e.g., masks, keypoints).
        max_det (int): Maximum detections to keep. Defaults to 300.
        conf_thres (float): Confidence threshold. Defaults to 0.25.
    Returns:
        torch.Tensor: Filtered detections of shape (*, 6 + n_extra).
    """
    score_start = 4
    score_end = 4 + nc
    score_view = pre_topk[:, score_start:score_end]
    ic = score_view.amax(dim=-1) > conf_thres
    pre_topk = pre_topk[ic]

    if pre_topk.shape[0] == 0:
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device)
    max_det = min(pre_topk.shape[0], max_det)

    row_index = torch.topk(pre_topk[:, score_start:score_end].amax(dim=-1), max_det, dim=0).indices
    selected = pre_topk[row_index]
    top_scores, flat_index = torch.topk(selected[:, score_start:score_end].reshape(-1), max_det)
    keep = top_scores > conf_thres
    if not torch.any(keep):
        return torch.zeros((0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device)

    top_scores = top_scores[keep]
    flat_index = flat_index[keep]
    box_index = flat_index // nc
    labels = (flat_index % nc).to(selected.dtype).unsqueeze(-1)

    output = torch.empty((top_scores.shape[0], 6 + n_extra), dtype=selected.dtype, device=selected.device)
    output[:, :4] = selected[box_index, :4]
    output[:, 4] = top_scores
    output[:, 5:6] = labels
    if n_extra > 0:
        output[:, 6:] = selected[box_index, score_end:]
    return output


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
            int((img1_shape[1] - img0_shape[1] * gain) / 2),
            int((img1_shape[0] - img0_shape[0] * gain) / 2),
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
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
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
    _, h, w = masks.shape
    out = torch.zeros_like(masks)
    boxes_i = torch.ceil(boxes).to(torch.int64)
    boxes_i[:, [0, 2]] = boxes_i[:, [0, 2]].clamp_(0, w)
    boxes_i[:, [1, 3]] = boxes_i[:, [1, 3]].clamp_(0, h)
    for i, (x1, y1, x2, y2) in enumerate(boxes_i.tolist()):
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return out


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
        pad_w, pad_h = (im1_w - im0_w * gain), (im1_h - im0_h * gain)  # wh padding
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
    img0_shapes: list[tuple[int, int]],
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
    labels_list: list[list[int]] = []
    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    for nms_out, img0_shape in zip(nms_outs, img0_shapes):
        boxes = nms_out[:, :4]
        scores = nms_out[:, 4]
        labels = nms_out[:, 5]
        boxes = scale_boxes(img1_shape, boxes, img0_shape)  # scale boxes to original image size
        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]  # xyxy to xywh with corner xy

        boxes_tolist = boxes.tolist()
        scores_tolist = scores.tolist()
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
    if isinstance(img0_shapes, tuple):
        actual_img0_shapes = [img0_shapes]
    else:
        actual_img0_shapes = img0_shapes

    if not isinstance(nms_outs[0], (list, tuple)):
        actual_nms_outs = [nms_outs]
    else:
        actual_nms_outs = nms_outs

    det_results = []
    seg_results = []
    for nms_out in actual_nms_outs:
        det_results.append(nms_out[0])
        seg_results.append(nms_out[1])

    labels_list, boxes_list, scores_list = nmsout2eval(det_results, img1_shape, actual_img0_shapes)

    scaled_seg_results = [
        scale_masks(seg_result.to(torch.float32), (img0_shape[0], img0_shape[1]))
        for seg_result, img0_shape in zip(seg_results, actual_img0_shapes)
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
) -> tuple[list[list[int]], list[list[list[float]]], list[list[float]], list[torch.Tensor]]:
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
    actual_img0_shapes = [img0_shapes] if isinstance(img0_shapes, tuple) else img0_shapes
    if not isinstance(nms_outs, list):
        actual_nms_outs = [nms_outs]
    else:
        actual_nms_outs = nms_outs
    labels_list, boxes_list, scores_list = nmsout2eval(actual_nms_outs, img1_shape, actual_img0_shapes)
    extra = [
        scale_coords(img1_shape, nms_out[:, 6:].reshape(-1, 17, 3), img0_shape).reshape(-1, 51)
        for nms_out, img0_shape in zip(actual_nms_outs, actual_img0_shapes)
    ]
    return labels_list, boxes_list, scores_list, extra


class YOLOSegPostMixin:
    """Mixin class for YOLO segmentation postprocessing."""

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[Any, ...]:
        """Converts NMS output to evaluation format for segmentation.

        Args:
            nms_out: NMS output (detections and prototypes).
            img1_shape: Resized image shape.
            img0_shape: List of original image shapes.

        Returns:
            Tuple: (labels_list, boxes_list, scores_list, extra_list).
        """
        return nmsout2eval_seg(nms_out, img1_shape, img0_shape)


class YOLOPosePostMixin:
    """Mixin class for YOLO pose estimation postprocessing."""

    def nmsout2eval(
        self,
        nms_out: Any,
        img1_shape: tuple[int, int],
        img0_shape: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[Any, ...]:
        """Converts NMS output to evaluation format for pose estimation.

        Args:
            nms_out: NMS output (detections with keypoints).
            img1_shape: Resized image shape.
            img0_shape: List of original image shapes.

        Returns:
            Tuple: (labels_list, boxes_list, scores_list, extra_list).
        """
        return nmsout2eval_pose(nms_out, img1_shape, img0_shape)
