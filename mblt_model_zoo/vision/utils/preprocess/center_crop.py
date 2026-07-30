"""
Center crop preprocessing.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from ._validation import normalize_image_size
from .base import PreOps


class CenterCrop(PreOps):
    """
    Center crop the image to a specified size.
    """

    def __init__(self, size: int | list[int] | tuple[int, int]) -> None:
        """Initializes the CenterCrop operation.

        Args:
            size (int | list[int]): Target size [h, w]. If int, size is [size, size].
        """
        super().__init__()
        self.size = normalize_image_size(size)

    def __call__(self, x: TensorLike | Image.Image) -> np.ndarray:
        """Applies center crop to the image.

        Args:
            x (np.ndarray | torch.Tensor | Image.Image): Input image.

        Returns:
            np.ndarray: Center-cropped image in HWC format.
        """
        if isinstance(x, torch.Tensor):
            image = x.cpu().numpy()
        elif isinstance(x, Image.Image):
            image = np.array(x)
        elif isinstance(x, np.ndarray):
            image = x
        else:
            raise TypeError(f"CenterCrop expects a NumPy array, tensor, or PIL image, got {type(x).__name__}.")
        if image.ndim != 3:
            raise ValueError(f"CenterCrop expects a three-dimensional image, got shape {image.shape}.")
        H, W = image.shape[:2]
        if (self.size[0] == H) and (self.size[1] == W):
            return image
        elif (self.size[1] > W) or (self.size[0] > H):
            image = cv2.copyMakeBorder(
                image,
                (self.size[0] - H) // 2 if self.size[0] > H else 0,
                (self.size[0] - H + 1) // 2 if self.size[0] > H else 0,
                (self.size[1] - W) // 2 if self.size[1] > W else 0,
                (self.size[1] - W + 1) // 2 if self.size[1] > W else 0,
                cv2.BORDER_CONSTANT,
                value=(0.0,),
            )
            H, W = image.shape[:2]
        crop_top = round((H - self.size[0]) / 2.0)
        crop_left = round((W - self.size[1]) / 2.0)
        image = image[
            crop_top : crop_top + self.size[0],
            crop_left : crop_left + self.size[1],
            :,
        ]
        return image.astype(np.uint8)
