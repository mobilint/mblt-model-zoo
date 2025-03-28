import numpy as np
import os
import cv2
import torch
from typing import List, Tuple, Union
from pycocotools.mask import encode


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


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
