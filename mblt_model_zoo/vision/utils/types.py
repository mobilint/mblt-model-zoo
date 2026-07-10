"""
Type definitions for MBLT vision models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch

TensorLike: TypeAlias = torch.Tensor | np.ndarray
ListTensorLike: TypeAlias = Sequence[TensorLike]
NestedListTensorLike: TypeAlias = Sequence[TensorLike | ListTensorLike]
