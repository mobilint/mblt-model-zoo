from typing import Any

from transformers.configuration_utils import (
    PretrainedConfig,
    SpecificPretrainedConfigType,
)

from .base_utils import MobilintNPUBackend


class MobilintConfigMixin(PretrainedConfig):
    def __init__(
        self,
        **kwargs
    ):
        self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")
        super().__init__(**kwargs)
    
    @property
    def mxq_path(self): return self.npu_backend.mxq_path
    @mxq_path.setter
    def mxq_path(self, v): self.npu_backend.mxq_path = v

    @property
    def dev_no(self): return self.npu_backend.dev_no
    @dev_no.setter
    def dev_no(self, v): self.npu_backend.dev_no = v

    @property
    def core_mode(self): return self.npu_backend.core_mode
    @core_mode.setter
    def core_mode(self, v): self.npu_backend.core_mode = v
    
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


class MobilintEncoderDecoderConfigMixin(PretrainedConfig):
    def __init__(
        self,
        **kwargs
    ):
        self.encoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="encoder_")
        self.decoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="decoder_")
        super().__init__(**kwargs)
    
    @property
    def encoder_mxq_path(self): return self.encoder_npu_backend.mxq_path
    @encoder_mxq_path.setter
    def encoder_mxq_path(self, v): self.encoder_npu_backend.mxq_path = v

    @property
    def encoder_dev_no(self): return self.encoder_npu_backend.dev_no
    @encoder_dev_no.setter
    def encoder_dev_no(self, v): self.encoder_npu_backend.dev_no = v

    @property
    def encoder_core_mode(self): return self.encoder_npu_backend.core_mode
    @encoder_core_mode.setter
    def encoder_core_mode(self, v): self.encoder_npu_backend.core_mode = v
    
    @property
    def encoder_target_cores(self): return self.encoder_npu_backend.target_cores
    @encoder_target_cores.setter
    def encoder_target_cores(self, v): self.encoder_npu_backend.target_cores = v
    
    @property
    def decoder_mxq_path(self): return self.decoder_npu_backend.mxq_path
    @decoder_mxq_path.setter
    def decoder_mxq_path(self, v): self.decoder_npu_backend.mxq_path = v

    @property
    def decoder_dev_no(self): return self.decoder_npu_backend.dev_no
    @decoder_dev_no.setter
    def decoder_dev_no(self, v): self.decoder_npu_backend.dev_no = v

    @property
    def decoder_core_mode(self): return self.decoder_npu_backend.core_mode
    @decoder_core_mode.setter
    def decoder_core_mode(self, v): self.decoder_npu_backend.core_mode = v
    
    @property
    def decoder_target_cores(self): return self.decoder_npu_backend.target_cores
    @decoder_target_cores.setter
    def decoder_target_cores(self, v): self.decoder_npu_backend.target_cores = v
    
    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "encoder_npu_backend"):
            _ = d.pop("encoder_npu_backend", None)
        
        if hasattr(self, "decoder_npu_backend"):
            _ = d.pop("decoder_npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.encoder_npu_backend.to_dict(prefix="encoder_"))
        output.update(self.decoder_npu_backend.to_dict(prefix="decoder_"))
        return output