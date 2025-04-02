from collections import OrderedDict
from .base import PreBase
from .resize import Resize
from .center_crop import CenterCrop
from .normalize import Normalize
from .order import SetOrder
from .reader import Reader
from .yolo_pre import YoloPre


def build_preprocess(pre_cfg: OrderedDict) -> PreBase:
    res = []
    for pre_type, pre_attr in pre_cfg.items():
        pre_type_lower = pre_type.lower()
        if pre_type_lower == Reader.__name__.lower():
            res.append(Reader(**pre_attr))
        elif pre_type_lower == Resize.__name__.lower():
            res.append(Resize(**pre_attr))
        elif pre_type_lower == CenterCrop.__name__.lower():
            res.append(CenterCrop(**pre_attr))
        elif pre_type_lower == Normalize.__name__.lower():
            res.append(Normalize(**pre_attr))
        elif pre_type_lower == SetOrder.__name__.lower():
            res.append(SetOrder(**pre_attr))
        elif pre_type_lower == YoloPre.__name__.lower():
            res.append(YoloPre(**pre_attr))
        else:
            raise ValueError(f"Got unsupported pre_type={pre_type}.")

    return PreBase(res)
