"""Configuration for Mobilint Qwen3 EAGLE-3 models."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ...utils.configuration_utils import MobilintEagle3ConfigMixin


class MobilintQwen3Eagle3Config(MobilintEagle3ConfigMixin, Qwen3Config):
    """Top-level config for Mobilint Qwen3 EAGLE-3.

    The base backend follows the Qwen3 architecture. The draft backend is a
    Llama-family 1-block draft trained separately, so ``draft_config`` is a
    raw ``LlamaConfig``. NPU backend fields for base/draft/fc live on the
    top-level config via the shared EAGLE-3 mixin; the draft sub-config only
    carries architecture-shape metadata.
    """

    model_type = "mobilint-qwen3-eagle3"
    sub_configs = {"draft_config": LlamaConfig}

    @classmethod
    def _get_draft_config_class(cls) -> type[PretrainedConfig]:
        return LlamaConfig

    def __init__(self, draft_config: dict[str, Any] | LlamaConfig | None = None, **kwargs: Any) -> None:
        if draft_config is not None:
            kwargs["draft_config"] = draft_config
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.draft_config.tie_word_embeddings = False


AutoConfig.register("mobilint-qwen3-eagle3", MobilintQwen3Eagle3Config)
