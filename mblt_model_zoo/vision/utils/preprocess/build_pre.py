"""
Preprocessing builder.
"""

from collections import OrderedDict

from .base import PreBase
from .center_crop import CenterCrop
from .letterbox import LetterBox
from .order import SetOrder
from .reader import Reader
from .resize import Resize


def build_preprocess(pre_cfg: OrderedDict) -> PreBase:
    """Builds a preprocessing pipeline based on the configuration.

    Args:
        pre_cfg (OrderedDict): Preprocessing configuration mapping operations to attributes.

    Returns:
        PreBase: An orchestrator for the sequence of preprocessing steps.
    """
    res = []
    for pre_type, pre_attr in pre_cfg.items():
        pre_type_lower = pre_type.lower()
        if pre_type_lower == Reader.__name__.lower():
            res.append(Reader(**pre_attr))
        elif pre_type_lower == Resize.__name__.lower():
            res.append(Resize(**pre_attr))
        elif pre_type_lower == CenterCrop.__name__.lower():
            res.append(CenterCrop(**pre_attr))
        elif pre_type_lower == SetOrder.__name__.lower():
            res.append(SetOrder(**pre_attr))
        elif pre_type_lower == LetterBox.__name__.lower():
            res.append(LetterBox(**pre_attr))
        else:
            raise ValueError(f"Got unsupported pre_type={pre_type}.")

    return PreBase(res)
