"""
Type definitions for MBLT vision models.
"""

from typing import List, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]
ListTensorLike = List[TensorLike]
