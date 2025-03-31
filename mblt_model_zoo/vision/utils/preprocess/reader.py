import PIL
import numpy as np
import PIL
import cv2
import torch
import os
from typing import Union
from .base import PreBase
from mblt_model_zoo.vision.utils.types import TensorLike


class Reader(PreBase):
    def __init__(self, style: str):
        """Read image and convert to tensor"""
        assert style in [
            "pil",
            "numpy",
        ], f"Unsupported style={style} for image reader."

        self.style = style

    def __call__(self, x: Union[str, TensorLike, PIL.Image.Image]):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if isinstance(x, torch.Tensor):
            if self.style == "pil":
                x = x.numpy()
                x = PIL.Image.fromarray(x.astype(np.uint8))
                return x
            else:
                return x

        elif isinstance(x, PIL.Image.Image):
            if self.style == "pil":
                return x
            else:
                return np.array(x)

        elif isinstance(x, str):
            assert os.path.exists(x) and os.path.isfile(x), f"File {x} does not exist."
            if self.style == "pil":
                return PIL.Image.open(x).convert("RGB")
            else:
                x = cv2.imread(x)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(x).float()

        else:
            raise ValueError(f"Unsupported input type={type(x)} for image reader.")
