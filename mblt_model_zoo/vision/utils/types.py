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
    This class is used to store model information.
    """

    pre_cfg: OrderedDict
    post_cfg: OrderedDict
    model_cfg: OrderedDict


class ModelInfoSet(Enum):
    """
    This class is used to store model informations.
    """
