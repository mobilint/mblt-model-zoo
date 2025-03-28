from .base import PostBase
import torch
import numpy as np


class ClsPost(PostBase):
    def __init__(self, pre_cfg: dict, post_cfg: dict):
        super().__init__()

    def __call__(self, x):
        x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.softmax(dim=-1)
            return x
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x).squeeze()
            x = x.softmax(dim=-1)
            return x.numpy()
        else:
            raise ValueError(f"Got unexpected type for x={type(x)}.")
