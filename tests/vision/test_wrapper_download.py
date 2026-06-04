"""Tests for vision wrapper MXQ path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
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
        if "subfolder" in kwargs:
            subfolder = kwargs["subfolder"]
            calls.append(subfolder)
            if subfolder == "aries":
                raise EntryNotFoundError("missing")
            return "/tmp/global8.mxq"
        else:
            filename = kwargs["filename"]
            assert filename == "model.onnx"
            return "/tmp/model.onnx"

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
    assert engine.file_cfg["onnx_filename"] == "model.onnx"
    assert engine.file_cfg["onnx_path"] == "/tmp/model.onnx"


def test_file_config_cleansing_resolves_local_onnx(
    tmp_path: Path,
) -> None:
    """Resolve ONNX file path next to local MXQ file when they exist locally."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

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
    assert engine.file_cfg["onnx_filename"] == "model.onnx"
    assert engine.file_cfg["onnx_path"] == str(onnx_path)


def test_prepare_onnx_inputs_keeps_batched_nchw_layout() -> None:
    """Preserve existing NCHW batches when feeding ONNX sessions."""

    class _FakeInput:
        name = "input"
        shape = [1, 3, 224, 224]

    class _FakeSession:
        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    engine.model = _FakeSession()
    engine.input_name = "input"

    batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)

    inputs = engine._prepare_onnx_inputs(batch)

    assert set(inputs) == {"input"}
    assert inputs["input"].shape == (2, 3, 224, 224)
    assert inputs["input"].dtype == np.float32


def test_prepare_onnx_inputs_transposes_hwc_images() -> None:
    """Convert single HWC images to batched NCHW arrays for ONNX runtime."""

    class _FakeInput:
        name = "input"
        shape = [1, 3, 224, 224]

    class _FakeSession:
        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    engine.model = _FakeSession()
    engine.input_name = "input"

    image = np.zeros((224, 224, 3), dtype=np.float32)

    inputs = engine._prepare_onnx_inputs(image)

    assert inputs["input"].shape == (1, 3, 224, 224)
