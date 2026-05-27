"""Configuration for Mobilint Qwen2 EAGLE-3 models."""

from __future__ import annotations

from typing import Any

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ...utils.configuration_utils import MobilintConfigMixin, MobilintEagle3ConfigMixin


class MobilintEagle3DraftConfig(MobilintConfigMixin, Qwen2Config):
    """Draft-side torch config for Mobilint EAGLE-3."""

    model_type = "mobilint-qwen2-eagle3-draft"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tie_word_embeddings = False


class MobilintQwen2Eagle3Config(MobilintEagle3ConfigMixin, Qwen2Config):
    """Top-level config for Mobilint Qwen2 EAGLE-3."""

    model_type = "mobilint-qwen2-eagle3"
    sub_configs = {"draft_config": MobilintEagle3DraftConfig}

    def __init__(self, draft_config: dict[str, Any] | MobilintEagle3DraftConfig | None = None, **kwargs: Any) -> None:
        eagle3_tree_depth = kwargs.pop("eagle3_tree_depth", None)
        eagle3_tree_top_k = kwargs.pop("eagle3_tree_top_k", None)
        eagle3_npu_chunk_size = kwargs.pop("eagle3_npu_chunk_size", None)
        draft_config_data = draft_config if draft_config is not None else kwargs.pop("draft_config", None)
        if draft_config_data is None:
            draft_config_data = {}
        if isinstance(draft_config_data, MobilintEagle3DraftConfig):
            self.draft_config = draft_config_data
        else:
            self.draft_config = MobilintEagle3DraftConfig(**draft_config_data)

        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.eagle3_tree_depth = int(eagle3_tree_depth if eagle3_tree_depth is not None else 5)
        self.eagle3_tree_top_k = int(eagle3_tree_top_k if eagle3_tree_top_k is not None else 8)
        self.eagle3_npu_chunk_size = int(eagle3_npu_chunk_size if eagle3_npu_chunk_size is not None else 192)
        self.draft_config.name_or_path = self.name_or_path


AutoConfig.register("mobilint-qwen2-eagle3", MobilintQwen2Eagle3Config)
