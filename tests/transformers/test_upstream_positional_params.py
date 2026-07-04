"""Unit tests for ``upstream_positional_params``."""

from __future__ import annotations

import functools

from mblt_model_zoo.hf_transformers.utils.generation_utils import (
    _upstream_positional_params_cached,
    upstream_positional_params,
)


def test_upstream_positional_params_filters_self_args_kwargs_and_kw_only() -> None:
    def f(self, a, b, *args, c=None, **kwargs):  # noqa: ANN001, ANN202
        del self, a, b, args, c, kwargs

    assert upstream_positional_params(f) == ("a", "b")


def test_upstream_positional_params_includes_positional_only() -> None:
    def f(self, a, /, b, *, c):  # noqa: ANN001, ANN202
        del self, a, b, c

    assert upstream_positional_params(f) == ("a", "b")


def test_upstream_positional_params_all_keyword_only_returns_empty() -> None:
    def f(self, *args, x, y):  # noqa: ANN001, ANN202
        del self, args, x, y

    assert upstream_positional_params(f) == ()


def test_upstream_positional_params_is_cached() -> None:
    def f(self, a, b):  # noqa: ANN001, ANN202
        del self, a, b

    assert upstream_positional_params(f) is upstream_positional_params(f)


def test_upstream_positional_params_wrapped_shares_cache_entry_with_underlying() -> None:
    def f(self, a, b):  # noqa: ANN001, ANN202
        del self, a, b

    @functools.wraps(f)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return f(*args, **kwargs)

    _upstream_positional_params_cached.cache_clear()
    underlying = upstream_positional_params(f)
    info_after_first = _upstream_positional_params_cached.cache_info()
    wrapped = upstream_positional_params(wrapper)
    info_after_second = _upstream_positional_params_cached.cache_info()

    assert underlying == wrapped
    assert info_after_first.misses == 1
    assert info_after_second.misses == 1
    assert info_after_second.hits == info_after_first.hits + 1
