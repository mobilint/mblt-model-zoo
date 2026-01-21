import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from maccel import Accelerator, Cluster, Core, CoreId, Model, ModelConfig

from ...utils.logging import log_model_details

if TYPE_CHECKING:
    MixinBase = Any
else:
    MixinBase = object
    
class PretrainedOnlyMixin(MixinBase):
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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs["_internal_call"] = True
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class MobilintNPUBackend:
    def __init__(
        self,
        name_or_path: str = "",
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, 'CoreId']]] = None,
        **kwargs
    ):
        self.name_or_path = name_or_path
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        self.core_mode = core_mode
        
        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []
    
    def launch(self):
        self.acc = Accelerator(self.dev_no)
        mc = ModelConfig()
        
        if self.core_mode == "single":
            mc.set_single_core_mode(0, self.target_cores)
        elif self.core_mode == "multi":
            mc.set_multi_core_mode(self.target_clusters)
        elif self.core_mode == "global4":
            mc.set_global4_core_mode(self.target_clusters)
        elif self.core_mode == "global8":
            assert len(self.target_clusters) == 2, "global8 must contain every cores!"
            mc.set_global8_core_mode()
        else:
            raise ValueError("core_mode must be single, multi, global4 or global8! value: " + self.core_mode)
        
        model_path = os.path.join(self.name_or_path, self.mxq_path)
        self.mxq_model = Model(model_path, mc)
        log_model_details(model_path)
        self.mxq_model.launch(self.acc)
    
    def dispose(self):
        self.mxq_model.dispose()

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
    
    @property
    def target_clusters(self) -> List['Cluster']:
        num_of_clusters = 2
        num_of_cores_in_cluster = 4
        
        core_id_lists = [[core_id for core_id in self.target_cores if core_id.cluster.value == i] for i in range(num_of_clusters)]
        
        for core_ids in core_id_lists:
            if len(core_ids) != num_of_cores_in_cluster and len(core_ids) != 0:
                raise ValueError(f"Target cores must include every cores in a cluster! core_ids: {self._target_cores_serialized}")
        
        return [core_ids[0].cluster for core_ids in core_id_lists if len(core_ids) == num_of_cores_in_cluster]

    def to_dict(self, prefix="") -> Dict[str, Any]:
        p = prefix
        return {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self.dev_no,
            f"{p}core_mode": self.core_mode,
            f"{p}target_cores": self._target_cores_serialized
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> 'MobilintNPUBackend':
        p = prefix
        return cls(
            name_or_path=data.get("name_or_path", ""),
            mxq_path=data.get(f"{p}mxq_path", ""),
            dev_no=data.get(f"{p}dev_no", 0),
            core_mode=data.get(f"{p}core_mode", "single"),
            target_cores=data.get(f"{p}target_cores", None)
        )
