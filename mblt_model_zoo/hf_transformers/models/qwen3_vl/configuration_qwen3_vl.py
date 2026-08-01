from functools import wraps

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from ...utils.configuration_utils import (
    MobilintConfigMixin,
    MobilintVisionTextConfigMixin,
)


class MobilintQwen3VLVisionConfig(MobilintConfigMixin, Qwen3VLVisionConfig):
    model_type = "mobilint-qwen3_vl"

    @wraps(Qwen3VLVisionConfig.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MobilintQwen3VLTextConfig(MobilintConfigMixin, Qwen3VLTextConfig):
    model_type = "mobilint-qwen3_vl_text"

    @wraps(Qwen3VLTextConfig.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MobilintQwen3VLConfig(MobilintVisionTextConfigMixin, Qwen3VLConfig):
    model_type = "mobilint-qwen3_vl"
    sub_configs = {"vision_config": MobilintQwen3VLVisionConfig, "text_config": MobilintQwen3VLTextConfig}

    def __init__(self, dynamic_vision: bool = False, **kwargs):
        # ``dynamic_vision`` pairs the vision MXQ, text MXQ, image processor,
        # and video processor as one release-level bundle. Nesting it under
        # ``vision_config`` would misleadingly frame it as a vision-only
        # property, so it lives at the top level. Guard against JSON
        # roundtrips (or upstream ordering changes) that would surface flat
        # ``text_*`` / ``vision_*`` NPU keys during
        # ``PretrainedConfig.__init__``, which would trigger the prefixed
        # property setters before sub-configs are available.
        text_kwargs, vision_kwargs = self._split_sub_backend_kwargs(kwargs)
        Qwen3VLConfig.__init__(self, **kwargs)
        self._apply_sub_backend_kwargs(text_kwargs, vision_kwargs)

        self.tie_word_embeddings = False
        self._attn_implementation = "eager"
        self.dynamic_vision = bool(dynamic_vision)


AutoConfig.register("mobilint-qwen3_vl", MobilintQwen3VLConfig)
