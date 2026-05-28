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


def test_qwen2_eagle3_config_exposes_prefixed_backend_properties() -> None:
    """EAGLE-3 config should expose base/draft/fc backend properties consistently."""
    config = MobilintQwen2Eagle3Config(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        base_core_mode="single",
        draft_core_mode="global4",
        fc_core_mode="global8",
        base_target_cores=["0:0"],
        draft_target_cores=["0:1"],
        fc_target_cores=["0:2"],
        base_target_clusters=[0],
        draft_target_clusters=[1],
        fc_target_clusters=[0, 1],
        draft_config={
            "vocab_size": 10,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )

    assert config.base_core_mode == "single"
    assert config.draft_core_mode == "global4"
    assert config.fc_core_mode == "global8"
    assert len(config.base_target_cores) == 1
    assert len(config.draft_target_cores) == 1
    assert len(config.fc_target_cores) == 1
    assert str(config.base_target_cores[0])
    assert str(config.draft_target_cores[0])
    assert str(config.fc_target_cores[0])
    assert len(config.base_target_clusters) == 1
    assert len(config.draft_target_clusters) == 1
    assert len(config.fc_target_clusters) == 2
    assert str(config.base_target_clusters[0])
    assert str(config.draft_target_clusters[0])
    assert all(str(cluster) for cluster in config.fc_target_clusters)


def test_qwen2_eagle3_config_name_or_path_propagates_to_draft_config() -> None:
    """Setting name_or_path on parent config should propagate to nested draft config."""
    config = MobilintQwen2Eagle3Config(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        draft_config={
            "vocab_size": 10,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )

    config.name_or_path = "mobilint/EAGLE3-JPharmatron-7B"

    assert config.draft_config.name_or_path == "mobilint/EAGLE3-JPharmatron-7B"
