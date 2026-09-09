from functools import wraps
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import (
    MobilintConfigMixin,
    _apply_npu_backend_kwargs,
    _split_npu_backend_kwargs,
)
from ._errors import guard_qwen_asr_import

with guard_qwen_asr_import():
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
    )


# Fields consumed by ``MobilintNPUBackend.from_dict`` that ``MobilintConfigMixin``
# does not expose as a forwarding property. Routing them through
# ``setattr(sub_config, k, v)`` would leave them on ``config.__dict__`` and
# never reach the backend, so both the constructor and ``from_dict`` override
# have to write them straight onto ``sub_config.npu_backend``. The value maps
# the caller-facing kwarg name to the backend's storage-name — ``commit_hash``
# is exposed publicly but stored as ``_commit_hash`` on the backend.
_QWEN3_ASR_BACKEND_DIRECT_FIELDS: dict[str, str] = {
    "revision": "revision",
    "commit_hash": "_commit_hash",
}


class MobilintQwen3ASRAudioEncoderConfig(MobilintConfigMixin, Qwen3ASRAudioEncoderConfig):
    model_type = "mobilint-qwen3_asr_audio_encoder"

    @wraps(Qwen3ASRAudioEncoderConfig.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MobilintQwen3ASRTextConfig(MobilintConfigMixin, Qwen3ASRTextConfig):
    model_type = "mobilint-qwen3_asr_text"

    @wraps(Qwen3ASRTextConfig.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MobilintQwen3ASRThinkerConfig(Qwen3ASRThinkerConfig):
    model_type = "mobilint-qwen3_asr_thinker"
    sub_configs = {
        "audio_config": MobilintQwen3ASRAudioEncoderConfig,
        "text_config": MobilintQwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id: int = 151646,
        audio_start_token_id: int = 151647,
        audio_end_token_id: int = 151648,
        user_token_id: int = 872,
        assistant_token_id: int = 77091,
        asr_text_token_id: int = 151704,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        newline_token_id: int = 198,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        PretrainedConfig.__init__(self, **kwargs)

        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id
        self.asr_text_token_id = asr_text_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.newline_token_id = newline_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.initializer_range = initializer_range
        self.audio_token_id = audio_token_id

        if isinstance(audio_config, dict):
            audio_config = MobilintQwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = MobilintQwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = MobilintQwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = MobilintQwen3ASRTextConfig()
        self.text_config = text_config


class MobilintQwen3ASRConfig(Qwen3ASRConfig):
    model_type = "mobilint-qwen3_asr"
    sub_configs = {
        "thinker_config": MobilintQwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        # Route `encoder_*` / `decoder_*` kwargs to the nested audio / text
        # NPU backends so callers get the same UI as Whisper's
        # MobilintEncoderDecoderConfigMixin.
        # encoder_* / decoder_* kwargs 를 thinker 의 audio / text sub-config
        # NPU backend 로 전달한다. Whisper 의 MobilintEncoderDecoderConfigMixin
        # 과 같은 사용자 인터페이스를 nested 구조에서 제공.
        encoder_kwargs = {
            k.removeprefix("encoder_"): kwargs.pop(k)
            for k in list(kwargs)
            if k.startswith("encoder_")
        }
        decoder_kwargs = {
            k.removeprefix("decoder_"): kwargs.pop(k)
            for k in list(kwargs)
            if k.startswith("decoder_")
        }

        PretrainedConfig.__init__(self, **kwargs)

        if isinstance(thinker_config, dict):
            thinker_config = MobilintQwen3ASRThinkerConfig(**thinker_config)
        elif thinker_config is None:
            thinker_config = MobilintQwen3ASRThinkerConfig()

        # Fields with a config-level property setter on ``MobilintConfigMixin``
        # go through the sub-config so ``target_device`` triggers the
        # cross-board rebuild via ``_rebuild_backend_for_target_device``
        # (setting the string on the backend directly would leave an
        # Aries instance in place with a Regulus target_device string).
        # Fields with no forwarding property
        # (:data:`_QWEN3_ASR_BACKEND_DIRECT_FIELDS`) route straight to the
        # nested backend, mapping ``commit_hash`` to its ``_commit_hash``
        # storage name.
        for sub_config, prefixed_kwargs in (
            (thinker_config.audio_config, encoder_kwargs),
            (thinker_config.text_config, decoder_kwargs),
        ):
            for k, v in prefixed_kwargs.items():
                if k in _QWEN3_ASR_BACKEND_DIRECT_FIELDS:
                    setattr(
                        sub_config.npu_backend,
                        _QWEN3_ASR_BACKEND_DIRECT_FIELDS[k],
                        v,
                    )
                else:
                    setattr(sub_config, k, v)

        self.thinker_config = thinker_config
        self.support_languages = support_languages

        # MXQ uses integer-quantized weights; float attention kernels (sdpa,
        # flash-attn) bring no benefit, so pin attention to eager.
        # MXQ 는 정수 양자화 가중치라 sdpa / flash-attn 같은 float attention
        # 경로가 의미 없다. eager 로 고정.
        self._attn_implementation = "eager"

    # Expose `encoder_*` / `decoder_*` as direct attributes so
    # `transformers.PretrainedConfig.from_pretrained` recognizes them via
    # `hasattr` and routes them to the nested NPU backends, instead of
    # leaking through to the model constructor.
    # encoder_* / decoder_* 를 config 의 직접 속성으로 노출. transformers
    # 의 from_pretrained 가 hasattr 로 config 필드 인식 후 nested NPU
    # backend 로 전달되도록 한다 (모델 인자로 새지 않음).

    @property
    def encoder_mxq_path(self) -> str:
        return self.thinker_config.audio_config.npu_backend.mxq_path

    @encoder_mxq_path.setter
    def encoder_mxq_path(self, value: str) -> None:
        self.thinker_config.audio_config.npu_backend.mxq_path = value

    @property
    def encoder_target_device(self) -> str:
        """Board identifier used by the audio encoder NPU backend."""
        return self.thinker_config.audio_config.npu_backend.target_device

    @encoder_target_device.setter
    def encoder_target_device(self, value: str) -> None:
        # Route through the sub-config property setter so cross-board
        # switches rebuild the backend via
        # ``_rebuild_backend_for_target_device`` instead of leaving an
        # Aries instance with a Regulus string.
        self.thinker_config.audio_config.target_device = value

    @property
    def encoder_dev_no(self) -> int:
        return self.thinker_config.audio_config.npu_backend.dev_no

    @encoder_dev_no.setter
    def encoder_dev_no(self, value: int) -> None:
        self.thinker_config.audio_config.npu_backend.dev_no = value

    @property
    def encoder_core_mode(self) -> str:
        return self.thinker_config.audio_config.npu_backend.core_mode

    @encoder_core_mode.setter
    def encoder_core_mode(self, value: str) -> None:
        self.thinker_config.audio_config.npu_backend.core_mode = value

    @property
    def encoder_max_batch_size(self) -> int:
        return self.thinker_config.audio_config.npu_backend.max_batch_size

    @encoder_max_batch_size.setter
    def encoder_max_batch_size(self, value: int) -> None:
        self.thinker_config.audio_config.npu_backend.max_batch_size = value

    @property
    def encoder_target_cores(self) -> list:
        return self.thinker_config.audio_config.npu_backend.target_cores

    @encoder_target_cores.setter
    def encoder_target_cores(self, values: list) -> None:
        self.thinker_config.audio_config.npu_backend.target_cores = values

    @property
    def encoder_target_clusters(self) -> list:
        return self.thinker_config.audio_config.npu_backend.target_clusters

    @encoder_target_clusters.setter
    def encoder_target_clusters(self, values: list) -> None:
        self.thinker_config.audio_config.npu_backend.target_clusters = values

    @property
    def decoder_mxq_path(self) -> str:
        return self.thinker_config.text_config.npu_backend.mxq_path

    @decoder_mxq_path.setter
    def decoder_mxq_path(self, value: str) -> None:
        self.thinker_config.text_config.npu_backend.mxq_path = value

    @property
    def decoder_target_device(self) -> str:
        """Board identifier used by the text decoder NPU backend."""
        return self.thinker_config.text_config.npu_backend.target_device

    @decoder_target_device.setter
    def decoder_target_device(self, value: str) -> None:
        # Route through the sub-config property setter (see the
        # ``encoder_target_device`` counterpart).
        self.thinker_config.text_config.target_device = value

    @property
    def decoder_dev_no(self) -> int:
        return self.thinker_config.text_config.npu_backend.dev_no

    @decoder_dev_no.setter
    def decoder_dev_no(self, value: int) -> None:
        self.thinker_config.text_config.npu_backend.dev_no = value

    @property
    def decoder_core_mode(self) -> str:
        return self.thinker_config.text_config.npu_backend.core_mode

    @decoder_core_mode.setter
    def decoder_core_mode(self, value: str) -> None:
        self.thinker_config.text_config.npu_backend.core_mode = value

    @property
    def decoder_max_batch_size(self) -> int:
        return self.thinker_config.text_config.npu_backend.max_batch_size

    @decoder_max_batch_size.setter
    def decoder_max_batch_size(self, value: int) -> None:
        self.thinker_config.text_config.npu_backend.max_batch_size = value

    @property
    def decoder_target_cores(self) -> list:
        return self.thinker_config.text_config.npu_backend.target_cores

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.thinker_config.text_config.npu_backend.target_cores = values

    @property
    def decoder_target_clusters(self) -> list:
        return self.thinker_config.text_config.npu_backend.target_clusters

    @decoder_target_clusters.setter
    def decoder_target_clusters(self, values: list) -> None:
        self.thinker_config.text_config.npu_backend.target_clusters = values

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs: Any):
        """Buffer ``encoder_*`` / ``decoder_*`` NPU kwargs across HF's ``from_dict``.

        This facade does not inherit ``MobilintEncoderDecoderConfigMixin``, so
        the shared buffered ``from_dict`` there does not apply here. Without
        buffering, HF's upstream ``PretrainedConfig.from_dict`` probes every
        override with ``hasattr`` before ``setattr``; the encoder / decoder
        topology getters exposed on this facade route through the nested
        ``npu_backend._spec`` finalize, so a caller override that combines
        ``encoder_target_device`` with a board-specific mode
        (e.g. ``encoder_core_mode="global8"`` on a Regulus baseline) raises
        during the probe before the target-device rebuild has a chance to
        run. Extract the NPU fields ahead of ``super().from_dict`` and
        replay them via ``_apply_npu_backend_kwargs`` in the shared apply
        order — ``target_device`` first — so the whole override set lands
        on the destination board atomically.

        Fields without a top-level forwarding property on this facade
        (``revision``, ``commit_hash``) cannot ride the shared setattr
        helper — ``setattr(config, "encoder_revision", v)`` would land on
        ``config.__dict__`` and leave the nested backend's own
        ``revision`` / ``_commit_hash`` untouched, so remote MXQ
        resolution would silently keep the shipped revision. Route those
        two directly to the nested backend the same way the constructor
        does (with ``commit_hash`` mapped to the internal
        ``_commit_hash`` storage name).
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        encoder_sub = _split_npu_backend_kwargs(kwargs, prefix="encoder_")
        decoder_sub = _split_npu_backend_kwargs(kwargs, prefix="decoder_")
        encoder_direct = {
            k: encoder_sub.pop(k)
            for k in list(encoder_sub)
            if k in _QWEN3_ASR_BACKEND_DIRECT_FIELDS
        }
        decoder_direct = {
            k: decoder_sub.pop(k)
            for k in list(decoder_sub)
            if k in _QWEN3_ASR_BACKEND_DIRECT_FIELDS
        }

        config, unused_kwargs = super().from_dict(
            config_dict, return_unused_kwargs=True, **kwargs
        )  # type: ignore[misc]

        _apply_npu_backend_kwargs(config, encoder_sub, prefix="encoder_")
        _apply_npu_backend_kwargs(config, decoder_sub, prefix="decoder_")
        for k, v in encoder_direct.items():
            setattr(
                config.thinker_config.audio_config.npu_backend,
                _QWEN3_ASR_BACKEND_DIRECT_FIELDS[k],
                v,
            )
        for k, v in decoder_direct.items():
            setattr(
                config.thinker_config.text_config.npu_backend,
                _QWEN3_ASR_BACKEND_DIRECT_FIELDS[k],
                v,
            )

        if return_unused_kwargs:
            return config, unused_kwargs
        return config


AutoConfig.register("mobilint-qwen3_asr", MobilintQwen3ASRConfig)
