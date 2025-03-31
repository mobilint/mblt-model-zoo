import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
from pycocotools.mask import encode
from mblt_model_zoo.vision.utils.datasets import get_coco_inv


def xywh2xyxy(x: Union[np.ndarray, torch.Tensor]):
    # Convert bounding box coordinates from (cx, cy, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    if isinstance(x, np.ndarray):
        y = np.copy(x)
    elif isinstance(x, torch.Tensor):
        y = torch.clone(x)
    else:
        raise ValueError("x should be np.ndarray or torch.Tensor")

    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def xyxy2xywh(x: Union[np.ndarray, torch.Tensor]):
    # Convert bounding box coordinates from (x1, y1, x2, y2) format to (cx, cy, width, height) format where (cx, cy) is the center of the bounding box and width and height are the dimensions of the bounding box.
    if isinstance(x, np.ndarray):
        y = np.copy(x)
    elif isinstance(x, torch.Tensor):
        y = torch.clone(x)
    else:
        raise ValueError("x should be np.ndarray or torch.Tensor")

    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]

    return y


def compute_ratio_pad(img1_shape, img0_shape, ratio_pad=None):
    """Compute ratio and pad which were used to resize image to input_shape"""
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


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def clip_boxes(boxes, shape):
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    return boxes


def clip_coords(coords, shape):
    """Clip coordinates to image shape."""
    coords[..., 0] = coords[..., 0].clamp(0, shape[1])
    coords[..., 1] = coords[..., 1].clamp(0, shape[0])
    return coords


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


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L680
    Faster way to process masks.
    protos: [mask_dim, mask_h, mask_w]
    masks_in: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape: input_image_size, (h, w)

    return: h, w, n
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
    masks = masks.gt_(0.0)

    return masks


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
    but is slower.

    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L713
    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        (torch.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # n, CHW
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def crop_mask(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks.
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
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


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


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
    order = torch.arange(scores.size(0))

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
        keep = ratio <= iou_threshold
        order = order[keep]

    return picked_indices


def nmsout2eval(nms_out, img1_shape, img0_shape):
    """NMS output to evaluation format.

    Args:
        nms_out (torch.Tensor): The output of the NMS operation of shape (n, 6), where n is the number of objects
        img1_shape (torch.Tensor): processed image shape
        img0_shape (torch.Tensor): original image shape

    Returns:
        labels (list): The labels of the objects
        boxes (list): The bounding boxes of the objects
        scores (list): The confidence scores of the objects
    """
    boxes = nms_out[:, :4]
    scores = nms_out[:, 4]
    labels = nms_out[:, 5]

    scale_boxes(img1_shape, boxes, img0_shape)
    boxes = xyxy2xywh(boxes)
    boxes[:, :2] -= boxes[:, 2:] / 2

    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()
    labels = [get_coco_inv(int(l)) for l in labels]

    return labels, boxes, scores


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
