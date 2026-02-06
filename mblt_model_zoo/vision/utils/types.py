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


class ModelInfoSet(Enum):
    """
    Enum for model information sets.
    """
