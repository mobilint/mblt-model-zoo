"""
Type definitions for MBLT vision models.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]
ListTensorLike = List[TensorLike]


@dataclass
class ModelInfo:
    """
    Data class for storing model configurations.

    Attributes:
        pre_cfg (OrderedDict): Preprocessing configuration.
        post_cfg (OrderedDict): Postprocessing configuration.
        model_cfg (OrderedDict): Model configuration.
    """

    pre_cfg: OrderedDict
    post_cfg: OrderedDict
    model_cfg: OrderedDict

    def update_model_cfg(self, **kwargs) -> "ModelInfo":
        """Returns a new ModelInfo with updated model_cfg while preserving OrderedDict."""
        new_model_cfg = OrderedDict(self.model_cfg)
        new_model_cfg.update(kwargs)
        return ModelInfo(
            pre_cfg=self.pre_cfg,
            post_cfg=self.post_cfg,
            model_cfg=new_model_cfg,
        )

    def update_pre_cfg(self, **kwargs) -> "ModelInfo":
        """Returns a new ModelInfo with updated pre_cfg while preserving OrderedDict."""
        new_pre_cfg = OrderedDict(self.pre_cfg)
        new_pre_cfg.update(kwargs)
        return ModelInfo(
            pre_cfg=new_pre_cfg,
            post_cfg=self.post_cfg,
            model_cfg=self.model_cfg,
        )


class ModelInfoSet(Enum):
    """
    Enum for model information sets.
    """
