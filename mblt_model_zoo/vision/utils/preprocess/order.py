import numpy as np
import torch
from .base import PreOps
from ..types import TensorLike


class SetOrder(PreOps):
    """Set the channel order of the image."""

    def __init__(self, shape: str = "HWC"):
        """Set the channel order of the image.

        Args:
            shape (str, optional): Channel order. Defaults to "HWC".
        """
        super().__init__()
        assert shape.lower() in ["hwc", "chw"], f"Got unsupported shape={shape}."
        self.shape = shape

    def __call__(self, x: TensorLike):
        assert x.ndim == 3, "Assume that x is a color image"
        if x.shape[0] == 3:
            cdim = 0
        elif x.shape[-1] == 3:
            cdim = 2
        else:
            raise ValueError(
                f"Only assume HWC or CHW with 3 channels, but got shape {x.shape}"
            )

        is_tensor = isinstance(x, torch.Tensor)
        permute_fn = torch.permute if is_tensor else np.transpose

        if cdim == 0 and self.shape.lower() == "hwc":
            return permute_fn(x, (1, 2, 0))
        elif cdim == 2 and self.shape.lower() == "chw":
            return permute_fn(x, (2, 0, 1))

        return x
