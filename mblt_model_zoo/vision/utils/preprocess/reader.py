"""
Image reader preprocessing.
"""

from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from ..types import TensorLike
from .base import PreOps


class Reader(PreOps):
    """
    Reader for loading images from file paths or converting existing objects.
    Supports "pil" and "numpy" reading styles.
    """

    def __init__(self, style: str):
        """Initializes the Reader operation.

        Args:
            style (str): Reading style, either "pil" or "numpy".
        """
        super().__init__()
        assert style.lower() in [
            "pil",
            "numpy",
        ], f"Unsupported style={style} for image reader."
        self.style = style.lower()

    def __call__(
        self, x: Union[str, TensorLike, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        """Reads/converts the input into an image object.

        Args:
            x (Union[str, TensorLike, Image.Image]): Input image path or image object.

        Returns:
            Union[np.ndarray, Image.Image]: Read image in the specified style.
        """
        if self.style == "numpy":
            if isinstance(x, np.ndarray):
                return x
            elif isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, str):
                x = cv2.imread(x)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                return x
            elif isinstance(x, Image.Image):
                return np.array(x)
            else:
                raise ValueError("Got Unsupported Input")
        elif self.style == "pil":
            if isinstance(x, np.ndarray):
                return Image.fromarray(x.astype(np.uint8))
            elif isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
                return Image.fromarray(x.astype(np.uint8))
            elif isinstance(x, str):
                return Image.open(x).convert("RGB")
            elif isinstance(x, Image.Image):
                return x
            else:
                raise ValueError("Got Unsupported Input")
        else:
            raise NotImplementedError("Got Unsupported Style")
