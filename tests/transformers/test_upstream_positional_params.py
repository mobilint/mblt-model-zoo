"""Unit tests for ``upstream_positional_params``."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.utils.generation_utils import upstream_positional_params


def test_upstream_positional_params_filters_self_args_and_kwargs() -> None:
    def f(self, a, b, *args, c=None, **kwargs):  # noqa: ANN001, ANN202
        del self, a, b, args, c, kwargs

    assert upstream_positional_params(f) == ("a", "b", "c")
