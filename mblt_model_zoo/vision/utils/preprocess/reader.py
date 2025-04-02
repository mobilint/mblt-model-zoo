import numpy as np
import cv2
import torch
from PIL import Image
from typing import Union
from .base import PreOps
from ..types import TensorLike


class Reader(PreOps):
    def __init__(self, style: str):
        """Read image and convert to tensor"""
        assert style.lower() in [
            "pil",
            "numpy",
        ], f"Unsupported style={style} for image reader."

        self.style = style.lower()

    def __call__(self, x: Union[str, TensorLike, Image.Image]):
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
