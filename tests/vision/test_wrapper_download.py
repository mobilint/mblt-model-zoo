"""Tests for vision wrapper MXQ path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import EntryNotFoundError

import mblt_model_zoo.vision.wrapper as wrapper
from mblt_model_zoo.vision.wrapper import MBLT_Engine


def test_file_config_cleansing_prefers_existing_mxq_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use an existing local MXQ path without attempting a Hub download."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")

    def _unexpected_download(**kwargs: Any) -> str:
        raise AssertionError("hf_hub_download should not be called")

    monkeypatch.setattr(wrapper, "hf_hub_download", _unexpected_download)

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.file_cfg = {
        "mxq_path": str(mxq_path),
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert engine.file_cfg["mxq_path"] == str(mxq_path)
    assert "repo_id" not in engine.file_cfg
    assert "filename" not in engine.file_cfg
    assert "revision" not in engine.file_cfg


def test_file_config_cleansing_downloads_aries_before_core_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Try the legacy aries layout before the core-mode-specific layout."""

    calls: list[str] = []

    def _fake_download(**kwargs: Any) -> str:
        subfolder = kwargs["subfolder"]
        calls.append(subfolder)
        if subfolder == "aries":
            raise EntryNotFoundError("missing")
        return "/tmp/global8.mxq"

    monkeypatch.setattr(wrapper, "hf_hub_download", _fake_download)

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.file_cfg = {
        "mxq_path": "",
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert calls == ["aries", "aries/global8"]
    assert engine.file_cfg["mxq_path"] == "/tmp/global8.mxq"
