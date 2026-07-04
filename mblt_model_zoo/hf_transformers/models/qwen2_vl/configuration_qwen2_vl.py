from functools import wraps

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLTextConfig,
    Qwen2VLVisionConfig,
)

from ...utils.configuration_utils import (
    MobilintConfigMixin,
    MobilintVisionTextConfigMixin,
)


class MobilintQwen2VLVisionConfig(MobilintConfigMixin, Qwen2VLVisionConfig):
    model_type = "mobilint-qwen2_vl"


class MobilintQwen2VLTextConfig(MobilintConfigMixin, Qwen2VLTextConfig):
    model_type = "mobilint-qwen2_vl_text"

    @wraps(Qwen2VLTextConfig.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MobilintQwen2VLConfig(MobilintVisionTextConfigMixin, Qwen2VLConfig):
    model_type = "mobilint-qwen2_vl"
    sub_configs = {"vision_config": MobilintQwen2VLVisionConfig, "text_config": MobilintQwen2VLTextConfig}

    @wraps(Qwen2VLConfig.__init__)
    def __init__(self, **kwargs):
        # Qwen2VLConfig calls `super().__init__(**kwargs)` before creating the
        # sub-configs, which drives PretrainedConfig into setattr'ing unknown
        # kwargs — that triggers our `text_*` / `vision_*` property setters
        # before `self.text_config` / `self.vision_config` exist. Strip and
        # re-apply after sub-configs are constructed.
        text_kwargs, vision_kwargs = self._split_sub_backend_kwargs(kwargs)
        Qwen2VLConfig.__init__(self, **kwargs)
        self._apply_sub_backend_kwargs(text_kwargs, vision_kwargs)

        self.tie_word_embeddings = False
        self._attn_implementation = "eager"


AutoConfig.register("mobilint-qwen2_vl", MobilintQwen2VLConfig)
