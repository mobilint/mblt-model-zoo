import os
from typing import Optional, Union

from qbruntime import Cluster, Core
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import logging

logger = logging.get_logger(__name__)

cluster_map = {
    0: Cluster.Cluster0,
    1: Cluster.Cluster1,
}

core_map = {
    0: Core.Core0,
    1: Core.Core1,
    2: Core.Core2,
    3: Core.Core3,
}

class PretrainedOnlyMixin(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        _internal_call = kwargs.pop("_internal_call", False)
        
        if not _internal_call:
            cls_name = self.__class__.__name__
            raise RuntimeError(
                f"Direct instantiation of {cls_name} is not allowed.\n"
                f"Please use `{cls_name}.from_pretrained(...)` to load the NPU model correctly."
            )
            
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        kwargs["_internal_call"] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
