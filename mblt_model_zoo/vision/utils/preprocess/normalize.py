import numpy as np
import torch
import torchvision.transforms as T
import PIL
from typing import Union
from .base import PreBase
from mblt_model_zoo.vision.utils.types import TensorLike


class Normalize(PreBase):
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

    def __call__(self, x: Union[TensorLike, PIL.Image.Image]):
        """Normalize image

        Args:
            x (np.ndarray): Image to be normalized.

        Returns:
            x: Normalized image.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, PIL.Image.Image):
            x = T.PILToTensor()(x).float()
        else:
            raise TypeError(f"Got unexpected type for x={type(x)})")

        if self.style == "torch":
            x = x / 255.0
            x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        elif self.style == "tf":
            x = x / 127.5 - 1.0
        return x
