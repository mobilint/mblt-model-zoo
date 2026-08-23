"""Unit tests for ``scripts/build_eagle3_safetensors.py``.

Covers the explicit-vs-defaulted ``--draft-subdir`` contract and the release-write-time ``d2t``
range check that guards against malformed draft checkpoints slipping onto disk as a "built"
release only to explode at candidate-sampling time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

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


def _make_release_fixture(
    tmp_path: Path,
    *,
    vocab_size: int,
    hidden_size: int,
    d2t_offsets: list[int],
) -> Path:
    """Assemble a minimal release folder with ``target_emb.pth``, ``draft_emb.pth``, and a draft safetensors.

    The embeddings are zero tensors of the requested shape so ``main()`` reaches the ``d2t`` range
    check without stumbling on unrelated dtype/shape guards.
    """
    source = tmp_path / "release"
    source.mkdir()
    target_emb = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
    torch.save(target_emb, source / "target_emb.pth")
    draft_emb = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
    torch.save(draft_emb, source / "draft_emb.pth")

    subdir = source / build.DEFAULT_DRAFT_SUBDIR
    subdir.mkdir()
    d2t = torch.tensor(d2t_offsets, dtype=torch.int64)
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    save_file({"d2t": d2t, "t2d": t2d}, str(subdir / "model.safetensors"))
    return source


def test_valid_d2t_writes_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed ``d2t`` (every ``i + d2t[i]`` in range) passes validation and writes the release."""
    source = _make_release_fixture(tmp_path, vocab_size=10, hidden_size=4, d2t_offsets=[5, 5, 5, 5])
    output = tmp_path / "release.safetensors"
    monkeypatch.setattr(sys, "argv", ["build_eagle3_safetensors.py", str(source), "--output", str(output)])

    assert build.main() == 0
    assert output.is_file()


def test_d2t_lower_bound_violation_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``i + d2t[i] < 0`` for any ``i`` must abort before writing the release."""
    # i=0: 0 + (-1) = -1 < 0
    source = _make_release_fixture(tmp_path, vocab_size=10, hidden_size=4, d2t_offsets=[-1, 5, 5, 5])
    output = tmp_path / "release.safetensors"
    monkeypatch.setattr(sys, "argv", ["build_eagle3_safetensors.py", str(source), "--output", str(output)])

    with pytest.raises(SystemExit) as excinfo:
        build.main()

    message = str(excinfo.value)
    assert "d2t" in message
    assert "i=0" in message
    assert "i + d2t[i]=-1" in message
    assert "vocab_size=10" in message
    assert not output.exists()


def test_d2t_upper_bound_violation_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``i + d2t[i] >= vocab_size`` for any ``i`` must abort before writing the release."""
    # i=3: 3 + 10 = 13 >= 10
    source = _make_release_fixture(tmp_path, vocab_size=10, hidden_size=4, d2t_offsets=[5, 5, 5, 10])
    output = tmp_path / "release.safetensors"
    monkeypatch.setattr(sys, "argv", ["build_eagle3_safetensors.py", str(source), "--output", str(output)])

    with pytest.raises(SystemExit) as excinfo:
        build.main()

    message = str(excinfo.value)
    assert "d2t" in message
    assert "i=3" in message
    assert "i + d2t[i]=13" in message
    assert "vocab_size=10" in message
    assert not output.exists()
