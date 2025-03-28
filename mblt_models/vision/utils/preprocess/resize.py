from typing import List, Union
import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
import PIL

from .base import PreBase
from mblt_models.vision.utils.types import *

TORCH_INTERP_CODES = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
}


class Resize(PreBase):
    """Resize image in Torch backend"""

    def __init__(
        self,
        size: Union[int, List[int]],
        interpolation: str,
    ):
        # Note that this behaves different for npy image and PIL image
        super().__init__()
        self.size = size  # h, w
        self.interpolation = interpolation

    def __call__(self, x: Union[TensorLike, PIL.Image.Image]):
        """Resize image

        Args:
            x (Union[np.ndarray, PIL.Image.Image]): Image to be resized.

        Raises:
            TypeError: If x is not numpy array or PIL image.

        Returns:
            x: Resized image.
        """
        # result: np.ndarray (H, W, C)

        if isinstance(x, np.ndarray):
            img_tensor = torch.from_numpy(x)
            resized_img_tensor = F.resize(
                img_tensor,
                size=self.size,
                interpolation=TORCH_INTERP_CODES[self.interpolation],
            )
            return resized_img_tensor.detach().numpy()
        elif isinstance(x, torch.Tensor) or isinstance(x, PIL.Image.Image):
            return F.resize(
                x,
                size=self.size,
                interpolation=TORCH_INTERP_CODES[self.interpolation],
            )
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")
