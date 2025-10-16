import torch
import numpy as np
from typing import Union
from .base import PostBase
from ..types import TensorLike, ListTensorLike


class ClsPost(PostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__()

    def __call__(self, x: Union[TensorLike, ListTensorLike]):
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
