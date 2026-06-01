"""Configuration for Mobilint Qwen2 EAGLE-3 models."""

from __future__ import annotations

from typing import Any

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.configuration_utils import PretrainedConfig

from ...utils.configuration_utils import MobilintEagle3ConfigMixin


class MobilintQwen2Eagle3Config(MobilintEagle3ConfigMixin, Qwen2Config):
    """Top-level config for Mobilint Qwen2 EAGLE-3."""

    model_type = "mobilint-qwen2-eagle3"
    sub_configs = {"draft_config": Qwen2Config}

    @classmethod
    def _get_draft_config_class(cls) -> type[PretrainedConfig]:
        return Qwen2Config

    def __init__(self, draft_config: dict[str, Any] | Qwen2Config | None = None, **kwargs: Any) -> None:
        if draft_config is not None:
            kwargs["draft_config"] = draft_config
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.draft_config.tie_word_embeddings = False


AutoConfig.register("mobilint-qwen2-eagle3", MobilintQwen2Eagle3Config)
