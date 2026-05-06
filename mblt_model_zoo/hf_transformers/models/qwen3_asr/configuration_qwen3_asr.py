from functools import wraps

from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRAudioEncoderConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import MobilintConfigMixin


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
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        PretrainedConfig.__init__(self, **kwargs)

        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
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
        PretrainedConfig.__init__(self, **kwargs)

        if isinstance(thinker_config, dict):
            thinker_config = MobilintQwen3ASRThinkerConfig(**thinker_config)
        elif thinker_config is None:
            thinker_config = MobilintQwen3ASRThinkerConfig()
        self.thinker_config = thinker_config
        self.support_languages = support_languages

        # MXQ weights are integer-quantized, so float dtype variants are irrelevant.
        # Force eager attention to bypass sdpa / flash-attn paths.
        self._attn_implementation = "eager"


AutoConfig.register("mobilint-qwen3_asr", MobilintQwen3ASRConfig)
