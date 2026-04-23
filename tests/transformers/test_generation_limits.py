"""Tests for transformer-only generation token limit policies."""

from __future__ import annotations

from tests.transformers.conftest import (
    transformers_batch_generation_token_limit,
    transformers_generation_token_limit,
)


class _FakeConfig:
    """Minimal pytest config stub for token-limit tests."""

    def __init__(self, *, full_matrix: bool):
        self._full_matrix = full_matrix

    def getoption(self, name: str) -> bool:
        """Return the stored pytest option value."""
        if name != "--full-matrix":
            raise KeyError(name)
        return self._full_matrix


def test_transformers_generation_token_limit_uses_quick_default():
    config = _FakeConfig(full_matrix=False)

    limit = transformers_generation_token_limit(config)

    assert limit == 32


def test_transformers_generation_token_limit_uses_full_matrix_value():
    config = _FakeConfig(full_matrix=True)

    limit = transformers_generation_token_limit(config)

    assert limit == 512


def test_transformers_batch_generation_token_limit_uses_quick_default():
    config = _FakeConfig(full_matrix=False)

    limit = transformers_batch_generation_token_limit(config)

    assert limit == 32


def test_transformers_batch_generation_token_limit_uses_full_matrix_value():
    config = _FakeConfig(full_matrix=True)

    limit = transformers_batch_generation_token_limit(config)

    assert limit == 256
