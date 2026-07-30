"""
Image reader preprocessing.
"""

from __future__ import annotations

from pathlib import Path

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

    def __init__(self, style: str) -> None:
        """Initializes the Reader operation.

        Args:
            style (str): Reading style, either "pil" or "numpy".
        """
        super().__init__()
        if not isinstance(style, str):
            raise TypeError(f"Reader style must be a string, got {type(style).__name__}.")
        if style.lower() not in {"pil", "numpy"}:
            raise ValueError(f"Unsupported Reader style {style!r}; expected 'pil' or 'numpy'.")
        self.style = style.lower()

    def __call__(self, x: str | Path | TensorLike | Image.Image) -> np.ndarray | Image.Image:
        """Reads/converts the input into an image object.

        Args:
            x (str | Path | TensorLike | Image.Image): Input image path or image object.

        Returns:
            np.ndarray | Image.Image: Read image in the specified style.
        """
        if self.style == "numpy":
            if isinstance(x, np.ndarray):
                return x
            elif isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, (str, Path)):
                image = cv2.imread(str(x))
                if image is None:
                    raise FileNotFoundError(f"Image not found: {x}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(x, Image.Image):
                return np.array(x)
            else:
                raise TypeError(f"Reader(style='numpy') does not support input type {type(x).__name__}.")
        elif self.style == "pil":
            if isinstance(x, np.ndarray):
                return Image.fromarray(x.astype(np.uint8))
            elif isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
                return Image.fromarray(x.astype(np.uint8))
            elif isinstance(x, (str, Path)):
                return Image.open(x).convert("RGB")
            elif isinstance(x, Image.Image):
                return x
            else:
                raise TypeError(f"Reader(style='pil') does not support input type {type(x).__name__}.")
        else:
            raise RuntimeError(f"Reader has an invalid validated style: {self.style!r}.")
