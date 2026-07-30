"""
Channel order preprocessing.
"""

from __future__ import annotations

import numpy as np
import torch

from ..types import TensorLike
from .base import PreOps


class SetOrder(PreOps):
    """Sets the channel order of the image to either HWC or CHW format."""

    def __init__(self, shape: str = "HWC") -> None:
        """Initializes the SetOrder operation.

        Args:
            shape (str, optional): Target channel order, either "HWC" or "CHW".
                Defaults to "HWC".
        """
        super().__init__()
        if not isinstance(shape, str):
            raise TypeError(f"SetOrder shape must be a string, got {type(shape).__name__}.")
        if shape.lower() not in {"hwc", "chw"}:
            raise ValueError(f"Unsupported channel order {shape!r}; expected 'HWC' or 'CHW'.")
        self.shape = shape.lower()

    def __call__(self, x: TensorLike) -> TensorLike:
        """Reorders the dimensions of the input image.

        Args:
            x (TensorLike): Input image of shape (3, H, W) or (H, W, 3).

        Returns:
            TensorLike: Image with the specified channel order.
        """
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(f"SetOrder expects a NumPy array or tensor, got {type(x).__name__}.")
        if x.ndim != 3:
            raise ValueError(f"SetOrder expects a three-dimensional color image, got shape {x.shape}.")
        channels_first = x.shape[0] == 3
        channels_last = x.shape[-1] == 3
        if channels_first and channels_last:
            raise ValueError(f"SetOrder cannot infer channel order from ambiguous shape {x.shape}.")
        if channels_first:
            cdim = 0
        elif channels_last:
            cdim = 2
        else:
            raise ValueError(f"SetOrder expects HWC or CHW data with three channels, got shape {x.shape}.")
        if cdim == 0 and self.shape == "hwc":
            if isinstance(x, torch.Tensor):
                return torch.permute(x, (1, 2, 0))
            return np.transpose(x, (1, 2, 0))
        elif cdim == 2 and self.shape == "chw":
            if isinstance(x, torch.Tensor):
                return torch.permute(x, (2, 0, 1))
            return np.transpose(x, (2, 0, 1))
        return x
