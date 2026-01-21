from typing import Any, Dict, List, Literal, Optional, Union

from maccel import Cluster, Core, CoreId
from transformers.configuration_utils import PretrainedConfig


class MobilintConfigMixin(PretrainedConfig):
    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, 'CoreId']]] = None,
        **kwargs
    ):
        _internal_call = kwargs.pop("_internal_call", False)
        
        if not _internal_call:
            config_class_name = self.__class__.__name__
            raise RuntimeError(
                f"Direct instantiation of {config_class_name} is not allowed.\n"
                f"Please use `{config_class_name}.from_pretrained(...)` to load the NPU model correctly."
            )
             
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        self.core_mode = core_mode
        
        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []

        super().__init__(**kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs["_internal_call"] = True
        
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @property
    def target_cores(self) -> List['CoreId']:
        result = []
        if not hasattr(self, "_target_cores_serialized"):
            return []
            
        for s in self._target_cores_serialized:
            try:
                c_val, r_val = map(int, s.split(':'))
                result.append(CoreId(Cluster(c_val), Core(r_val)))
            except Exception as e:
                pass
        return result

    @target_cores.setter
    def target_cores(self, values: List[Union[str, 'CoreId']]):
        serialized = []
        for v in values:
            if isinstance(v, CoreId):
                serialized.append(f"{v.cluster.value}:{v.core.value}")
            elif isinstance(v, str):
                if ":" in v:
                    serialized.append(v)
                else:
                    raise ValueError(f"Invalid format: {v}")
            else:
                raise TypeError(f"Unsupported type: {type(v)}")
        
        self._target_cores_serialized = serialized

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        
        if hasattr(self, "_target_cores_serialized"):
            output["target_cores"] = self._target_cores_serialized
        
        if "_target_cores_serialized" in output:
            del output["_target_cores_serialized"]
            
        return output
