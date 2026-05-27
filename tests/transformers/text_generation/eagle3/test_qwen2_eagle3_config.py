"""Unit tests for Mobilint Qwen2 EAGLE-3 config plumbing."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
    MobilintQwen2Eagle3Config,
)


def test_qwen2_eagle3_config_roundtrip_preserves_nested_draft_and_backend_fields() -> None:
    """EAGLE-3 config should round-trip nested draft config and backend overrides."""
    config = MobilintQwen2Eagle3Config(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        eagle3_tree_depth=6,
        eagle3_tree_top_k=4,
        eagle3_npu_chunk_size=128,
        draft_config={
            "vocab_size": 10,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )

    serialized = config.to_dict()
    restored = MobilintQwen2Eagle3Config(**serialized)

    assert restored.base_mxq_path == "base.mxq"
    assert restored.draft_mxq_path == "draft.mxq"
    assert restored.fc_mxq_path == "fc.mxq"
    assert restored.eagle3_tree_depth == 6
    assert restored.eagle3_tree_top_k == 4
    assert restored.eagle3_npu_chunk_size == 128
    assert restored.draft_config.vocab_size == 10
