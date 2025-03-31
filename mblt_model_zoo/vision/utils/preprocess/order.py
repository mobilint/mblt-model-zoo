import numpy as np
from typing import Union
from .base import *
import torch


class SetOrder(PreBase):
    """Set the channel order of the image."""

    def __init__(self, shape: str = "HWC"):
        """Set the channel order of the image.

        Args:
            shape (str, optional): Channel order. Defaults to "HWC".
        """
        super().__init__()
        assert shape.lower() in ["hwc", "chw"], f"Got unsupported shape={shape}."
        self.shape = shape

    def __call__(self, x: np.ndarray):
        # Assume x is HWC
        if isinstance(x, np.ndarray):
            if self.shape.lower() == "hwc":
                return x
            elif self.shape.lower() == "chw":
                x = x.transpose(2, 0, 1)
                return x
        elif isinstance(x, torch.Tensor):
            if self.shape.lower() == "hwc":
                return x.permute(1, 2, 0)
            elif self.shape.lower() == "chw":
                return x
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")
