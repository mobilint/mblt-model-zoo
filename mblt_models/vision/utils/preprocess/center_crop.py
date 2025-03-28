from typing import List, Union
import PIL
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from .base import PreBase
from mblt_models.vision.utils.types import TensorLike


class CenterCrop(PreBase):
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

    def __call__(self, x: Union[TensorLike, PIL.Image.Image]):
        if isinstance(x, np.ndarray):
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
        elif isinstance(x, torch.Tensor) or isinstance(x, PIL.Image.Image):
            return T.CenterCrop(self.size)(x)
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")
