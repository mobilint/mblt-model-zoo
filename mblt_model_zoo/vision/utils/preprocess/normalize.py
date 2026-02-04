"""
Normalize image preprocessing
"""

from typing import Union

import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from .base import PreOps


class Normalize(PreOps):
    """
    Normalize image pixel values.
    Supports "torch" (standard ImageNet mean/std) and "tf" (range [-1, 1]) styles.
    """

    def __init__(
        self,
        style: str = "torch",
    ):
        """Initializes the Normalize operation.

        Args:
            style (str): Normalization style. Supported values are "torch", "tf".
                Defaults to "torch".
        """
        super().__init__()
        self.style = style
        if self.style not in ["torch", "tf"]:
            raise ValueError(f"style {self.style} not supported.")

    def __call__(self, x: Union[TensorLike, Image.Image]) -> torch.Tensor:
        """Normalizes the input image.

        Args:
            x (Union[np.ndarray, torch.Tensor, Image.Image]): Image to be normalized.

        Returns:
            torch.Tensor: Normalized image as a float tensor on the target device.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, Image.Image):
            x = torch.from_numpy(np.array(x)).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            raise TypeError(f"Got unexpected type for x={type(x)})")
        if self.style == "torch":
            x = x / 255.0
            x -= torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(self.device)
            x /= torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(self.device)
        elif self.style == "tf":
            x = x / 127.5 - 1.0
        return x
