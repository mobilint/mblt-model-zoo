import copy
from typing import Any

from transformers.configuration_utils import (
    PretrainedConfig,
    SpecificPretrainedConfigType,
)

from .base_utils import MobilintNPUBackend


class MobilintConfigMixin(PretrainedConfig):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")
        super().__init__(*args, **kwargs)

    @property
    def mxq_path(self) -> str:
        return self.npu_backend.mxq_path

    @mxq_path.setter
    def mxq_path(self, value: str) -> None:
        self.npu_backend.mxq_path = value

    @property
    def dev_no(self) -> int:
        return self.npu_backend.dev_no

    @dev_no.setter
    def dev_no(self, value: int) -> None:
        self.npu_backend.dev_no = value

    @property
    def core_mode(self) -> str:
        return self.npu_backend.core_mode

    @core_mode.setter
    def core_mode(self, value: str) -> None:
        self.npu_backend.core_mode = value

    @property
    def target_cores(self) -> list:
        return self.npu_backend.target_cores

    @target_cores.setter
    def target_cores(self, values: list) -> None:
        self.npu_backend.target_cores = values

    @property
    def target_clusters(self) -> list:
        return self.npu_backend.target_clusters

    @target_clusters.setter
    def target_clusters(self, values: list) -> None:
        self.npu_backend.target_clusters = values
    
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
    def encoder_mxq_path(self) -> str:
        return self.encoder_npu_backend.mxq_path

    @encoder_mxq_path.setter
    def encoder_mxq_path(self, value: str) -> None:
        self.encoder_npu_backend.mxq_path = value

    @property
    def encoder_dev_no(self) -> int:
        return self.encoder_npu_backend.dev_no

    @encoder_dev_no.setter
    def encoder_dev_no(self, value: int) -> None:
        self.encoder_npu_backend.dev_no = value

    @property
    def encoder_core_mode(self) -> str:
        return self.encoder_npu_backend.core_mode

    @encoder_core_mode.setter
    def encoder_core_mode(self, value: str) -> None:
        self.encoder_npu_backend.core_mode = value

    @property
    def encoder_target_cores(self) -> list:
        return self.encoder_npu_backend.target_cores

    @encoder_target_cores.setter
    def encoder_target_cores(self, values: list) -> None:
        self.encoder_npu_backend.target_cores = values

    @property
    def encoder_target_clusters(self) -> list:
        return self.encoder_npu_backend.target_clusters

    @encoder_target_clusters.setter
    def encoder_target_clusters(self, values: list) -> None:
        self.encoder_npu_backend.target_clusters = values

    @property
    def decoder_mxq_path(self) -> str:
        return self.decoder_npu_backend.mxq_path

    @decoder_mxq_path.setter
    def decoder_mxq_path(self, value: str) -> None:
        self.decoder_npu_backend.mxq_path = value

    @property
    def decoder_dev_no(self) -> int:
        return self.decoder_npu_backend.dev_no

    @decoder_dev_no.setter
    def decoder_dev_no(self, value: int) -> None:
        self.decoder_npu_backend.dev_no = value

    @property
    def decoder_core_mode(self) -> str:
        return self.decoder_npu_backend.core_mode

    @decoder_core_mode.setter
    def decoder_core_mode(self, value: str) -> None:
        self.decoder_npu_backend.core_mode = value

    @property
    def decoder_target_cores(self) -> list:
        return self.decoder_npu_backend.target_cores

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.decoder_npu_backend.target_cores = values

    @property
    def decoder_target_clusters(self) -> list:
        return self.decoder_npu_backend.target_clusters

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.decoder_npu_backend.target_cores = values
        
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

    def get_text_config(self, decoder=None, encoder=None) -> "PretrainedConfig":
        return self
