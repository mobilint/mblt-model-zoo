"""Regression tests for the Qwen3-VL remote-code proxy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_proxy_module(proxy_path: Path) -> None:
    """Execute the proxy module from disk with an isolated module name."""
    spec = importlib.util.spec_from_file_location("tests.qwen3_vl_proxy_under_test", proxy_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_qwen3_vl_proxy_rejects_unsupported_transformers_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject Transformers versions older than the upstream Qwen3-VL integration."""
    proxy_path = (
        Path(__file__).resolve().parents[2]
        / "mblt_model_zoo"
        / "hf_transformers"
        / "models"
        / "qwen3_vl"
        / "proxy_qwen3_vl.py"
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "4.56.9"
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ImportError, match=r"transformers>=4\.57\.0"):
        _load_proxy_module(proxy_path)
