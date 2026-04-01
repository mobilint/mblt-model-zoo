"""Normalization operation for image preprocessing."""

from typing import Union

import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from .base import PreOps

STYLE_PARAMS = {
    "torch": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tf": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "openai": (
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ),
    "cv": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
STYLE_LIST = list(STYLE_PARAMS.keys())


class Normalize(PreOps):
    """Normalization layer to scale and shift image data.

    Attributes:
        style: Data source style (e.g., 'torch', 'tf', 'openai', 'cv').
        mean: Array of mean values for normalization.
        std: Array of standard deviation values for normalization.
    """

    def __init__(self, style: str):
        """Initializes the Normalize layer with a specific style.

        Args:
            style: The preprocessing style to use. Must be one of STYLE_LIST.
        """
        super().__init__()

        assert style.lower() in STYLE_LIST, f"Got unexpected style={style}. The style must be one of {STYLE_LIST}."

        self.style = style.lower()
        mean, std = STYLE_PARAMS[self.style]
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x: Union[TensorLike, Image.Image]) -> np.ndarray:
        """Applies normalization to the input image or tensor.

        Args:
            x: Input data as a torch.Tensor, PIL Image, or numpy-like array.

        Returns:
            The normalized image as a float32 numpy array.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        elif isinstance(x, Image.Image):
            x = np.array(x)
        x = x.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x
