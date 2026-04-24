"""
Channel order preprocessing.
"""

import numpy as np
import torch

from ..types import TensorLike
from .base import PreOps


class SetOrder(PreOps):
    """Sets the channel order of the image to either HWC or CHW format."""

    def __init__(self, shape: str = "HWC"):
        """Initializes the SetOrder operation.

        Args:
            shape (str, optional): Target channel order, either "HWC" or "CHW".
                Defaults to "HWC".
        """
        super().__init__()
        assert shape.lower() in ["hwc", "chw"], f"Got unsupported shape={shape}."
        self.shape = shape

    def __call__(self, x: TensorLike) -> TensorLike:
        """Reorders the dimensions of the input image.

        Args:
            x (TensorLike): Input image of shape (3, H, W) or (H, W, 3).

        Returns:
            TensorLike: Image with the specified channel order.
        """
        assert x.ndim == 3, "Assume that x is a color image"
        if x.shape[0] == 3:
            cdim = 0
        elif x.shape[-1] == 3:
            cdim = 2
        else:
            raise ValueError(f"Only assume HWC or CHW with 3 channels, but got shape {x.shape}")
        if cdim == 0 and self.shape.lower() == "hwc":
            if isinstance(x, torch.Tensor):
                return torch.permute(x, (1, 2, 0))
            return np.transpose(x, (1, 2, 0))
        elif cdim == 2 and self.shape.lower() == "chw":
            if isinstance(x, torch.Tensor):
                return torch.permute(x, (2, 0, 1))
            return np.transpose(x, (2, 0, 1))
        return x
