"""
Center crop preprocessing.
"""

from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from .base import PreOps


class CenterCrop(PreOps):
    """
    Center crop the image to a specified size.
    """

    def __init__(self, size: Union[int, List[int]]):
        """Initializes the CenterCrop operation.

        Args:
            size (Union[int, List[int]]): Target size [h, w]. If int, size is [size, size].
        """
        super().__init__()
        if isinstance(size, list):
            assert len(size) == 2, f"Got unexpected size={size}."
            self.size = size
        elif isinstance(size, int):
            self.size = [size, size]

    def __call__(self, x: Union[TensorLike, Image.Image]) -> np.ndarray:
        """Applies center crop to the image.

        Args:
            x (Union[np.ndarray, torch.Tensor, Image.Image]): Input image.

        Returns:
            np.ndarray: Center-cropped image in HWC format.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        elif isinstance(x, Image.Image):
            x = np.array(x)
        H, W = x.shape[:2]
        if (self.size[0] == H) and (self.size[1] == W):
            return x
        elif (self.size[1] > W) or (self.size[0] > H):
            x = cv2.copyMakeBorder(
                x,
                (self.size[0] - H) // 2 if self.size[0] > H else 0,
                (self.size[0] - H + 1) // 2 if self.size[0] > H else 0,
                (self.size[1] - W) // 2 if self.size[1] > W else 0,
                (self.size[1] - W + 1) // 2 if self.size[1] > W else 0,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            H, W = x.shape[:2]
        crop_top = round((H - self.size[0]) / 2.0)
        crop_left = round((W - self.size[1]) / 2.0)
        x = x[
            crop_top : crop_top + self.size[0],
            crop_left : crop_left + self.size[1],
            :,
        ]
        return x.astype(np.uint8)
