"""Unit tests for Mobilint Llama configuration validation."""

from __future__ import annotations

import pytest

from mblt_model_zoo.hf_transformers.models.llama.configuration_llama import MobilintLlamaConfig


def test_llama_config_validate_allows_kanana_non_divisible_hidden_size() -> None:
    """Kanana configs should remain valid after restoring their real hidden size."""
    config = MobilintLlamaConfig(
        hidden_size=1792,
        intermediate_size=4864,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=128,
    )

    config.validate()

    assert config.hidden_size == 1792
    assert config.num_attention_heads == 24
    assert config.head_dim == 128


def test_llama_config_validate_rejects_non_positive_head_dim() -> None:
    """Mobilint validation should still reject invalid explicit head dimensions."""
    config = MobilintLlamaConfig(
        hidden_size=1792,
        intermediate_size=4864,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=128,
    )
    config.head_dim = 0

    with pytest.raises(ValueError, match="head_dim must be positive"):
        config.validate()