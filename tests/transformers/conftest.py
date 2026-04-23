"""Shared pytest fixtures for transformer test suites."""

from __future__ import annotations

import pytest

from tests.npu_backend_options import full_matrix_enabled

QUICK_GENERATION_TOKEN_LIMIT = 32
FULL_MATRIX_GENERATION_TOKEN_LIMIT = 512


def transformers_generation_token_limit(config: pytest.Config) -> int:
    """Return the shared generation cap for quick and full transformer test runs."""
    if full_matrix_enabled(config):
        return FULL_MATRIX_GENERATION_TOKEN_LIMIT
    return QUICK_GENERATION_TOKEN_LIMIT


@pytest.fixture(scope="session")
def generation_token_limit(request: pytest.FixtureRequest) -> int:
    """Return the shared generation cap for transformer smoke tests."""
    return transformers_generation_token_limit(request.config)
