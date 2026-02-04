"""
Common postprocessing utility functions.
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


# --- Box Conversion Utilities ---
def xywh2xyxy(x: Union[np.ndarray, torch.Tensor]):
    """Converts bounding box coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input bounding boxes in (cx, cy, w, h) format.

    Returns:
        Union[np.ndarray, torch.Tensor]: Bounding boxes in (x1, y1, x2, y2) format.
    """
    if isinstance(x, np.ndarray):
        y = np.copy(x)
    elif isinstance(x, torch.Tensor):
        y = torch.clone(x).to(x)
    else:
        raise ValueError("x should be np.ndarray or torch.Tensor")
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: Union[np.ndarray, torch.Tensor]):
    """Converts bounding box coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    (cx, cy) is the center of the bounding box.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        Union[np.ndarray, torch.Tensor]: Bounding boxes in (cx, cy, w, h) format.
    """
    if isinstance(x, np.ndarray):
        y = np.copy(x)
    elif isinstance(x, torch.Tensor):
        y = torch.clone(x).to(x)
    else:
        raise ValueError("x should be np.ndarray or torch.Tensor")
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance (ltrb) to bounding box (xywh or xyxy).
    Args:
        distance (torch.Tensor): Distance from anchor points to box boundaries (left, top, right, bottom).
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
def non_max_suppression(boxes, scores, iou_threshold, max_output):
    """
    Modified non-maximum suppression (NMS) implemented with PyTorch.
    Args:
        boxes (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
        scores (torch.Tensor): Confidence scores for each box (assumed to be sorted in descending order).
        iou_threshold (float): IoU threshold for suppression.
        max_output (int): Maximum number of boxes to keep.
    Returns:
        List[int]: Indices of the boxes that have been kept after NMS.
    """
    if boxes.numel() == 0:
        return []
    # Coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    picked_indices = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x) * (end_y - start_y)
    # Create an index order (assumed scores are already sorted in descending order)
    order = torch.arange(scores.size(0)).to(boxes.device)
    while order.numel() > 0 and len(picked_indices) < max_output:
        # The index with the highest score
        index = order[0].item()
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
    if pre_topk.shape[0] == 0:
        return torch.zeros(
            (0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device
        )
    max_det = min(pre_topk.shape[0], max_det)
    # first topk
    box, scores, extra = pre_topk.split([4, nc, n_extra], dim=-1)
    max_scores = scores.amax(dim=-1)
    _, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    box = torch.gather(box, dim=0, index=index.expand(-1, 4))
    scores = torch.gather(scores, dim=0, index=index.expand(-1, nc))
    extra = torch.gather(extra, dim=0, index=index.expand(-1, n_extra))
    # second topk
    scores, index = torch.topk(scores.flatten(), max_det)
    index = index.unsqueeze(-1)
    scores = scores.unsqueeze(-1)
    labels = (index % nc).float()
    index = index // nc
    box = box.gather(dim=0, index=index.expand(-1, 4))
    extra = extra.gather(dim=0, index=index.expand(-1, n_extra))
    box_cls = torch.cat([box, scores, labels, extra], dim=1)  # (max_det, 6 + n_extra)
    box_cls = box_cls[box_cls[:, 4] > conf_thres]  # final filtering
    if box_cls.numel() == 0:
        return torch.zeros(
            (0, 6 + n_extra), dtype=torch.float32, device=pre_topk.device
        )
    return box_cls


# --- Scaling & Clipping Utilities ---
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Original Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L92
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).
    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (np.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    Returns:
        boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)
    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, padding=True):
    """
    Original Source:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L756
    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        coords (tuple): The coordinates of the objects in the image, in the format of (x, y).
        img0_shape (tuple): The shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    Returns:
        coords (tuple): The scaled coordinates, in the format of (x, y)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = int((img1_shape[1] - img0_shape[1] * gain) / 2), int(
            (img1_shape[0] - img0_shape[0] * gain) / 2
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., :2] /= gain
    return clip_coords(coords, img0_shape)


def compute_ratio_pad(img1_shape, img0_shape, ratio_pad=None):
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
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image shape.
    Args:
        boxes (torch.Tensor): Bounding boxes.
        shape (tuple): Image shape (height, width).
    Returns:
        torch.Tensor: Clipped bounding boxes.
    """
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    return boxes


def clip_coords(coords, shape):
    """Clips coordinates to the image shape.

    Args:
        coords (torch.Tensor): Coordinates to clip.
        shape (tuple): Image shape (height, width).

    Returns:
        torch.Tensor: Clipped coordinates.
    """
    coords[..., 0] = coords[..., 0].clamp(0, shape[1])
    coords[..., 1] = coords[..., 1].clamp(0, shape[0])
    return coords


# --- Segmentation Utilities ---
def process_mask(protos, masks_in, bboxes, shape, upsample=False):
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
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[
            0
        ]  # CHW
    return masks.gt_(0.0)


def process_mask_upsample(protos, masks_in, bboxes, shape):
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
    masks = scale_masks(masks, shape)  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def crop_mask(masks, boxes):
    """Crops masks to bounding boxes.

    Args:
        masks (torch.Tensor): Masks of shape [n, h, w].
        boxes (torch.Tensor): Bounding boxes of shape [n, 4] in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
        None, None, :
    ]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
        None, :, None
    ]  # cols shape(1,h,1)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
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
    masks = F.interpolate(
        masks[None], shape, mode="bilinear", align_corners=False
    )  # 1NHW
    return masks[0]
