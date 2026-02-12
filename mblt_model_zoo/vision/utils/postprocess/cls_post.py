"""
Classification postprocessing.
"""

from typing import Union

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .base import PostBase


class ClsPost(PostBase):
    """Post-processing for image classification models.

    Typically applies softmax to logits and ensures correct output shape.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict):
        """Initializes the classification post-processing.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        super().__init__()

    def __call__(self, x: Union[TensorLike, ListTensorLike]) -> torch.Tensor:
        """Executes classification post-processing.

        Typically applies softmax to convert logits to probabilities.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model outputs.
                Expected to be pre-softmax logits.

        Returns:
            torch.Tensor: Softmax probabilities of shape (N, C).
        """
        if isinstance(x, list):
            assert (
                len(x) == 1
            ), "assume that classification model only returns pre-softmax tensor"
            x = x[0]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        else:
            x = x.to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        assert (
            x.ndim == 4
        ), f"Assume that the result is always in form of NCHW. But the shape is {x.shape}"
        x = x.flatten(1)  # assume that the shape can be made to (b, 1000)
        return x.softmax(dim=-1)
