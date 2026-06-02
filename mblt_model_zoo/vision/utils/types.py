"""
Type definitions for MBLT vision models.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

TensorLike: TypeAlias = torch.Tensor | np.ndarray
ListTensorLike: TypeAlias = list[TensorLike]
