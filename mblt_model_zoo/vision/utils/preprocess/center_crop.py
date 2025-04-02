from typing import List, Union
from PIL import Image
import torch
import numpy as np
import cv2
from .base import PreOps
from ..types import TensorLike


class CenterCrop(PreOps):
    def __init__(self, size: Union[int, List[int]]):
        """Center crop the image

        Args:
            size (Union[int, List[int]]): Target height and width.
        """
        super().__init__()

        if isinstance(size, list):
            assert len(size) == 2, f"Got unexpected size={size}."
            self.size = size
        elif isinstance(size, int):
            self.size = [size, size]

    def __call__(self, x: Union[TensorLike, Image.Image]):
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

        return x
