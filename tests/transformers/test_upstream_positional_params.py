"""Unit tests for ``upstream_positional_params``."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.utils.generation_utils import upstream_positional_params


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
