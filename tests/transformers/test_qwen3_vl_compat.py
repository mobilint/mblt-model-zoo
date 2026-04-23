"""Unit tests for Qwen3-VL pytest compatibility helpers."""

from __future__ import annotations

import pytest

from tests.transformers import qwen3_vl_compat


def test_transformers_supports_qwen3_vl_returns_true_when_all_modules_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report support when every upstream Qwen3-VL module can be resolved."""
    monkeypatch.setattr(qwen3_vl_compat.importlib.util, "find_spec", lambda module_name: object())

    assert qwen3_vl_compat.transformers_supports_qwen3_vl() is True


def test_transformers_supports_qwen3_vl_returns_false_when_any_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report no support when any upstream Qwen3-VL module cannot be resolved."""

    def _fake_find_spec(module_name: str) -> object | None:
        if module_name.endswith("modeling_qwen3_vl"):
            return None
        return object()

    monkeypatch.setattr(qwen3_vl_compat.importlib.util, "find_spec", _fake_find_spec)

    assert qwen3_vl_compat.transformers_supports_qwen3_vl() is False


def test_skip_if_transformers_lacks_qwen3_vl_support_raises_module_level_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise pytest's skip exception when upstream Qwen3-VL support is unavailable."""
    monkeypatch.setattr(qwen3_vl_compat.importlib.util, "find_spec", lambda module_name: None)

    with pytest.raises(pytest.skip.Exception, match=r"Qwen3-VL classes"):
        qwen3_vl_compat.skip_if_transformers_lacks_qwen3_vl_support()
