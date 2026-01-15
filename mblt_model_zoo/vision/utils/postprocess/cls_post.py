"""Classification post-processing."""

from typing import Union

import numpy as np
import torch

from ..types import ListTensorLike, TensorLike
from .base import PostBase


class ClsPost(PostBase):
    """Classification post-processing class."""

    def __init__(self, _pre_cfg: dict, _post_cfg: dict):
        """
        Initialize the ClsPost class.

        Args:
            _pre_cfg (dict): Preprocessing configuration.
            _post_cfg (dict): Postprocessing configuration.
        """
        super().__init__()

    def __call__(self, x: Union[TensorLike, ListTensorLike]):
        """
        Apply softmax to classification outputs.

        Args:
            x (Union[TensorLike, ListTensorLike]): Raw model output.

        Returns:
            torch.Tensor: Softmax probabilities.
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
        ), f"Assume that the result is always in form of NHWC. But the shape is {x.shape}"

        x = x.flatten(1)  # assume that the shape can be made to (b, 1000)
        return x.softmax(dim=-1)
