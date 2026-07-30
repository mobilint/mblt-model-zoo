"""
Classification postprocessing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .base import PostBase


class ClsPost(PostBase):
    """Post-processing for image classification models.

    Typically applies softmax to logits and ensures correct output shape.
    """

    def __init__(self, pre_cfg: dict, post_cfg: dict) -> None:
        """Initializes the classification post-processing.

        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        super().__init__()
        self.softmax = post_cfg.get("softmax", False)

    def __call__(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Executes classification post-processing.

        Typically applies softmax to convert logits to probabilities.

        Args:
            x (TensorLike | ListTensorLike): Raw model outputs.
                Expected to be pre-softmax logits.

        Returns:
            torch.Tensor: Softmax probabilities of shape (N, C).
        """
        if isinstance(x, Sequence):
            if len(x) != 1:
                raise ValueError(f"Classification postprocessing expects one output tensor, got {len(x)}.")
            x = x[0]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Classification output must be convertible to NCHW, got shape {tuple(x.shape)}.")
        x = x.flatten(1)  # Classification heads may retain singleton spatial dimensions.
        if self.softmax:
            return x
        return x.softmax(dim=-1)
