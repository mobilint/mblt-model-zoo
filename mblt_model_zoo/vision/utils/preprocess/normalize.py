import numpy as np
import torch
from PIL import Image
from typing import Union
from .base import PreOps
from ..types import TensorLike


class Normalize(PreOps):
    def __init__(
        self,
        style: str = "torch",
    ):
        """Normalize image

        Args:
            style (str): Normalization style. Supported values are "torch", "tf"

        """
        super().__init__()
        self.style = style
        if self.style not in ["torch", "tf"]:
            raise ValueError(f"style {self.style} not supported.")

    def __call__(self, x: Union[TensorLike, Image.Image]):
        """Normalize image

        Args:
            x (np.ndarray): Image to be normalized.

        Returns:
            x: Normalized image.
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
