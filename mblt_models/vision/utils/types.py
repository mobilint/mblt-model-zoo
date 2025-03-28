from typing import Union, List, Any
import torch
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum

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

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj
