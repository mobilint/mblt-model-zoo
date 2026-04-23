import inspect
from inspect import Parameter, Signature
from typing import Any, TypeVar, Union

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.configuration_utils import SpecificPretrainedConfigType
except ImportError:
    try:
        from transformers.configuration_utils import SpecificPreTrainedConfigType as SpecificPretrainedConfigType
    except ImportError:
        SpecificPretrainedConfigType = TypeVar(
            "SpecificPretrainedConfigType",
            bound=PretrainedConfig,
        )

from ...utils.npu_backend import MobilintNPUBackend


class MobilintConfigMixin(PretrainedConfig):
    _NPU_SIGNATURE_FIELDS = (
        ("mxq_path", "", str),
        ("dev_no", 0, int),
        ("max_batch_size", 1, int),
        ("core_mode", "single", str),
        ("target_cores", None, Any),
        ("target_clusters", None, Any),
        ("revision", None, Any),
        ("npu_prefill_chunk_size", None, Any),
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._augment_init_signature()

    @classmethod
    def _augment_init_signature(cls) -> None:
        """Expose Mobilint backend kwargs to upstream config introspection."""
        init = cls.__init__
        signature = inspect.signature(init)
        if any(name in signature.parameters for name, _, _ in cls._NPU_SIGNATURE_FIELDS):
            return

        parameters = list(signature.parameters.values())
        insert_at = next(
            (index for index, parameter in enumerate(parameters) if parameter.kind == Parameter.VAR_KEYWORD),
            len(parameters),
        )
        extra_parameters = [
            Parameter(name=name, kind=Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
            for name, default, annotation in cls._NPU_SIGNATURE_FIELDS
        ]
        init.__signature__ = Signature(parameters[:insert_at] + extra_parameters + parameters[insert_at:])

    def _ensure_npu_backend(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "npu_backend"):
            self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")

    def __init__(self, *args, **kwargs):
        self._ensure_npu_backend(kwargs)
        super().__init__(*args, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_npu_backend(kwargs)
        super().__post_init__(**kwargs)

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
    def max_batch_size(self) -> int:
        return self.npu_backend.max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value: int) -> None:
        self.npu_backend.max_batch_size = max(1, value)

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

    @property
    def npu_prefill_chunk_size(self) -> Any:
        return self.__dict__.get("npu_prefill_chunk_size", None)

    @npu_prefill_chunk_size.setter
    def npu_prefill_chunk_size(self, value: Any) -> None:
        self.__dict__["npu_prefill_chunk_size"] = value

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "npu_backend"):
            _ = d.pop("npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self, "npu_backend"):
            output.update(self.npu_backend.to_dict(prefix=""))
        return output


class MobilintEncoderDecoderConfigMixin(PretrainedConfig):
    def _ensure_encoder_decoder_npu_backends(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "encoder_npu_backend"):
            self.encoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="encoder_")

        if not hasattr(self, "decoder_npu_backend"):
            self.decoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="decoder_")

    def __init__(self, **kwargs):
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__init__(**kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__post_init__(**kwargs)

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
    def encoder_max_batch_size(self) -> int:
        return self.encoder_npu_backend.max_batch_size

    @encoder_max_batch_size.setter
    def encoder_max_batch_size(self, value: int) -> None:
        self.encoder_npu_backend.max_batch_size = max(1, value)

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
    def decoder_max_batch_size(self) -> int:
        return self.decoder_npu_backend.max_batch_size

    @decoder_max_batch_size.setter
    def decoder_max_batch_size(self, value: int) -> None:
        self.decoder_npu_backend.max_batch_size = max(1, value)

    @property
    def decoder_target_cores(self) -> list:
        return self.decoder_npu_backend.target_cores

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.decoder_npu_backend.target_cores = values

    @property
    def decoder_target_clusters(self) -> list:
        return self.decoder_npu_backend.target_clusters

    @decoder_target_clusters.setter
    def decoder_target_clusters(self, values: list) -> None:
        self.decoder_npu_backend.target_clusters = values

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "encoder_npu_backend"):
            _ = d.pop("encoder_npu_backend", None)

        if hasattr(self, "decoder_npu_backend"):
            _ = d.pop("decoder_npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()

        if hasattr(self, "encoder_npu_backend"):
            output.update(self.encoder_npu_backend.to_dict(prefix="encoder_"))
        if hasattr(self, "decoder_npu_backend"):
            output.update(self.decoder_npu_backend.to_dict(prefix="decoder_"))

        return output

    def get_text_config(self, decoder=None, encoder=None) -> "PretrainedConfig":
        return self


class MobilintVisionTextConfigMixin(PretrainedConfig):
    sub_configs = {"vision_config": MobilintConfigMixin, "text_config": MobilintConfigMixin}

    @property
    def vision_mxq_path(self) -> str:
        return self.vision_config.mxq_path

    @vision_mxq_path.setter
    def vision_mxq_path(self, value: str) -> None:
        self.vision_config.mxq_path = value

    @property
    def vision_dev_no(self) -> int:
        return self.vision_config.dev_no

    @vision_dev_no.setter
    def vision_dev_no(self, value: int) -> None:
        self.vision_config.dev_no = value

    @property
    def vision_core_mode(self) -> str:
        return self.vision_config.core_mode

    @vision_core_mode.setter
    def vision_core_mode(self, value: str) -> None:
        self.vision_config.core_mode = value

    @property
    def vision_max_batch_size(self) -> int:
        return self.vision_config.max_batch_size

    @vision_max_batch_size.setter
    def vision_max_batch_size(self, value: int) -> None:
        self.vision_config.max_batch_size = max(1, value)

    @property
    def vision_target_cores(self) -> list:
        return self.vision_config.target_cores

    @vision_target_cores.setter
    def vision_target_cores(self, values: list) -> None:
        self.vision_config.target_cores = values

    @property
    def vision_target_clusters(self) -> list:
        return self.vision_config.target_clusters

    @vision_target_clusters.setter
    def vision_target_clusters(self, values: list) -> None:
        self.vision_config.target_clusters = values

    @property
    def text_mxq_path(self) -> str:
        return self.text_config.mxq_path

    @text_mxq_path.setter
    def text_mxq_path(self, value: str) -> None:
        self.text_config.mxq_path = value

    @property
    def text_dev_no(self) -> int:
        return self.text_config.dev_no

    @text_dev_no.setter
    def text_dev_no(self, value: int) -> None:
        self.text_config.dev_no = value

    @property
    def text_core_mode(self) -> str:
        return self.text_config.core_mode

    @text_core_mode.setter
    def text_core_mode(self, value: str) -> None:
        self.text_config.core_mode = value

    @property
    def text_max_batch_size(self) -> int:
        return self.text_config.max_batch_size

    @text_max_batch_size.setter
    def text_max_batch_size(self, value: int) -> None:
        self.text_config.max_batch_size = max(1, value)

    @property
    def text_target_cores(self) -> list:
        return self.text_config.target_cores

    @text_target_cores.setter
    def text_target_cores(self, values: list) -> None:
        self.text_config.target_cores = values

    @property
    def text_target_clusters(self) -> list:
        return self.text_config.target_clusters

    @text_target_clusters.setter
    def text_target_clusters(self, values: list) -> None:
        self.text_config.target_clusters = values

    @property
    def text_npu_prefill_chunk_size(self) -> Any:
        return self.text_config.npu_prefill_chunk_size

    @text_npu_prefill_chunk_size.setter
    def text_npu_prefill_chunk_size(self, value: Any) -> None:
        self.text_config.npu_prefill_chunk_size = value

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintVisionTextConfigMixin", tuple["MobilintVisionTextConfigMixin", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config: MobilintVisionTextConfigMixin
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)  # type: ignore

        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path

        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MobilintConfigMixin,
        vision_config: MobilintConfigMixin,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )
