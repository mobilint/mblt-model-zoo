"""Unit tests for ``scripts/build_eagle3_safetensors.py::_resolve_draft_subdir``.

The tests focus on the explicit-vs-defaulted ``--draft-subdir`` contract: a defaulted request
may auto-detect a single candidate, but an explicit request must exist.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_SCRIPT = _REPO_ROOT / "scripts" / "build_eagle3_safetensors.py"


def _load_build_module() -> ModuleType:
    """Load ``scripts/build_eagle3_safetensors.py`` as an isolated module.

    ``scripts/`` is not a Python package, so a file-loader keeps the test self-contained without
    polluting ``sys.path``.
    """
    spec = importlib.util.spec_from_file_location("build_eagle3_safetensors_under_test", _BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _make_subdir_with_safetensors(source: Path, name: str) -> Path:
    """Create ``source/<name>/model.safetensors`` as an empty flag file and return the subdir."""
    subdir = source / name
    subdir.mkdir(parents=True, exist_ok=False)
    (subdir / "model.safetensors").touch()
    return subdir


def test_default_subdir_present_is_used(tmp_path: Path) -> None:
    """When ``requested`` is ``None`` and the default subdir exists, return the default."""
    default_subdir = _make_subdir_with_safetensors(tmp_path, build.DEFAULT_DRAFT_SUBDIR)
    _make_subdir_with_safetensors(tmp_path, "other_ckpt")

    resolved = build._resolve_draft_subdir(tmp_path, None)

    assert resolved == default_subdir


def test_default_missing_auto_detects_single_candidate(tmp_path: Path) -> None:
    """When ``requested`` is ``None`` and the default is absent, adopt the single candidate."""
    only_candidate = _make_subdir_with_safetensors(tmp_path, "epoch_5_step_100000")

    resolved = build._resolve_draft_subdir(tmp_path, None)

    assert resolved == only_candidate


def test_explicit_missing_subdir_never_auto_detects(tmp_path: Path) -> None:
    """An explicit ``--draft-subdir`` that does not exist must fail even if a single candidate does."""
    _make_subdir_with_safetensors(tmp_path, "epoch_5_step_100000")

    with pytest.raises(SystemExit) as excinfo:
        build._resolve_draft_subdir(tmp_path, "typo_epoch")

    message = str(excinfo.value)
    assert "typo_epoch" in message
    assert "explicitly" in message
