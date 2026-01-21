from typing import Any

from transformers.configuration_utils import PretrainedConfig

from .base_utils import MobilintNPUBackend


class MobilintConfigMixin(PretrainedConfig):
    def __init__(
        self,
        **kwargs
    ):
        self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")
        super().__init__(**kwargs)

    @property
    def target_cores(self): return self.npu_backend.target_cores
    @target_cores.setter
    def target_cores(self, v): self.npu_backend.target_cores = v
    
    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "npu_backend"):
            _ = d.pop("npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.npu_backend.to_dict(prefix=""))
        return output