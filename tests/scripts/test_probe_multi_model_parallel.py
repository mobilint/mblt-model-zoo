"""Unit tests for the multi-model parallel-inference probe.

These tests exercise ``_run_for_n``'s slot-launch loop against a fake
``qbruntime.Model`` so the launch-failure dispose behavior can be verified
without touching real accelerators.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_PATH = _REPO_ROOT / "scripts" / "probe_multi_model_parallel.py"


def _load_probe_module() -> ModuleType:
    """Load ``scripts/probe_multi_model_parallel.py`` as an isolated module."""
    spec = importlib.util.spec_from_file_location("probe_multi_model_parallel_under_test", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()
_QBRUNTIME_ERROR = probe.QbRuntimeError


class _FakeAccelerator:
    """Minimal stand-in for ``qbruntime.Accelerator`` used by the probe."""

    def __init__(self, dev_no: int) -> None:
        self.dev_no = int(dev_no)


class _FakeModel:
    """Fake ``qbruntime.Model`` recording lifecycle events for verification.

    ``launch`` succeeds ``launch_ok_before_fail`` times before raising
    ``QbRuntimeError``. ``infer`` returns a fixed constant so the caller can
    build synthetic inputs from ``get_model_input_shape`` /
    ``get_model_input_data_type`` without hitting real hardware.
    """

    instances: list["_FakeModel"] = []
    launch_ok_before_fail: int = 1_000_000
    _launched_ok: int = 0

    @classmethod
    def _reset(cls, launch_ok_before_fail: int = 1_000_000) -> None:
        cls.instances = []
        cls.launch_ok_before_fail = int(launch_ok_before_fail)
        cls._launched_ok = 0

    def __init__(self, mxq_path: str, model_config: Any) -> None:
        self.mxq_path = mxq_path
        self.disposed = False
        self.launched = False
        _FakeModel.instances.append(self)

    def launch(self, acc: _FakeAccelerator) -> None:
        if _FakeModel._launched_ok >= _FakeModel.launch_ok_before_fail:
            raise _QBRUNTIME_ERROR("simulated launch BadAlloc")
        _FakeModel._launched_ok += 1
        self.launched = True

    def dispose(self) -> None:
        self.disposed = True

    def get_model_input_shape(self) -> list[list[int]]:
        return [[1, 4]]

    def get_model_input_data_type(self) -> str:
        return "DataType.Float32"

    def infer(self, inputs: list[Any]) -> list[Any]:
        import numpy as np

        return [np.zeros((1, 4), dtype=np.float32)]


def _args(n: int = 2) -> SimpleNamespace:
    """Build a minimal ``argparse``-shaped namespace for ``_run_for_n``."""
    return SimpleNamespace(
        mxq_path="/dev/null/fake.mxq",
        dev_no=0,
        core_mode="single",
        partition_cores=False,
        repeat=1,
        warmup=0,
        seq_len=1,
        seed=0,
        output_parity=False,
        n_models=str(n),
        output_dir=".",
    )


@pytest.fixture(autouse=True)
def _patch_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route Model/Accelerator to the in-process fakes for every test."""
    monkeypatch.setattr(probe, "Model", _FakeModel)
    monkeypatch.setattr(probe, "Accelerator", _FakeAccelerator)


def test_launch_failure_disposes_unappended_handle() -> None:
    """When ``mm.launch`` raises on slot 1, the orphan handle must be disposed.

    Regression guard: without the fix, ``mm`` is constructed (allocating LPDDR)
    but never appended to ``models``, so the outer ``finally``'s
    ``_dispose_all(models)`` cannot release it. ``main`` continues probing
    larger ``N`` on the same device after a per-``N`` failure, so the leak
    otherwise cascades into BadAlloc.
    """
    _FakeModel._reset(launch_ok_before_fail=1)

    with pytest.raises(_QBRUNTIME_ERROR):
        probe._run_for_n(n=2, args=_args(n=2), baseline_per_model_median=None)

    assert len(_FakeModel.instances) == 2
    first, orphan = _FakeModel.instances
    # Slot 0 launched successfully and was appended to ``models`` — the outer
    # ``finally: _dispose_all(models)`` disposes it in the normal cleanup path.
    assert first.launched is True
    assert first.disposed is True
    # Slot 1 failed inside ``launch`` before ever being appended. The on-the-spot
    # dispose (the fix) is the only reason its LPDDR gets released.
    assert orphan.launched is False
    assert orphan.disposed is True


def test_normal_path_appends_and_disposes_via_finally() -> None:
    """When every launch succeeds, all handles land in ``models`` and are disposed once."""
    _FakeModel._reset(launch_ok_before_fail=1_000_000)

    result = probe._run_for_n(n=2, args=_args(n=2), baseline_per_model_median=None)

    assert result["n_models"] == 2
    assert len(_FakeModel.instances) == 2
    for mm in _FakeModel.instances:
        assert mm.launched is True
        assert mm.disposed is True


class _ConstructFailModel:
    """Fake ``qbruntime.Model`` whose ``__init__`` raises without allocating a handle."""

    instances: list["_ConstructFailModel"] = []

    def __init__(self, mxq_path: str, model_config: Any) -> None:  # noqa: ARG002
        raise _QBRUNTIME_ERROR("simulated construct BadAlloc")

    def launch(self, acc: _FakeAccelerator) -> None:  # pragma: no cover - never called
        raise AssertionError("launch must not run when __init__ raises")

    def dispose(self) -> None:  # pragma: no cover - never called
        raise AssertionError("dispose must not run when there is no handle to release")


def test_construct_failure_no_dispose_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``Model(...)`` itself raises, there is nothing to dispose.

    The exception must surface out of ``_run_for_n`` unchanged and the
    launch-failure dispose branch must NOT be reached (no ``mm`` was ever
    bound to a live handle).
    """
    monkeypatch.setattr(probe, "Model", _ConstructFailModel)

    with pytest.raises(_QBRUNTIME_ERROR):
        probe._run_for_n(n=1, args=_args(n=1), baseline_per_model_median=None)


def test_speedup_is_none_when_baseline_missing() -> None:
    """With no baseline (``--n-models`` omits 1) ``_run_for_n`` returns ``speedup=None``.

    Regression guard: the previous fallback fabricated ``1.0``, which looked
    like a valid measurement of flat scaling and misled callers.
    """
    _FakeModel._reset(launch_ok_before_fail=1_000_000)

    row = probe._run_for_n(n=2, args=_args(n=2), baseline_per_model_median=None)

    assert row["speedup_vs_n1"] is None


def test_speedup_computed_when_baseline_present() -> None:
    """A concrete baseline produces the ``n * baseline / wall`` speedup formula."""
    _FakeModel._reset(launch_ok_before_fail=1_000_000)

    row = probe._run_for_n(n=2, args=_args(n=2), baseline_per_model_median=1.0)

    speedup = row["speedup_vs_n1"]
    wall_median = row["wall_time_s"]["median"]
    assert speedup is not None
    assert wall_median > 0
    assert speedup == pytest.approx(2 * 1.0 / wall_median)


def test_csv_renders_null_speedup_as_empty_string(tmp_path: Path) -> None:
    """CSV writer must emit an empty cell for a ``None`` speedup — matches ``parity_median``."""
    row = {
        "n_models": 2,
        "wall_time_s": {"median": 0.01, "min": 0.009, "max": 0.011},
        "per_model_time_s": {"median": 0.008, "min": 0.007, "max": 0.009},
        "n_batches": 1,
        "throughput_infers_per_s": 200.0,
        "speedup_vs_n1": None,
        "parity_median": None,
    }
    csv_path = tmp_path / "probe_report.csv"

    probe._write_csv([row], csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1]
    speedup_idx = header.index("speedup_vs_n1")
    parity_idx = header.index("parity_median")
    assert data[speedup_idx] == ""
    assert data[parity_idx] == ""


def test_csv_renders_valid_speedup(tmp_path: Path) -> None:
    """CSV writer preserves the 6-decimal format for a valid speedup."""
    row = {
        "n_models": 2,
        "wall_time_s": {"median": 0.01, "min": 0.009, "max": 0.011},
        "per_model_time_s": {"median": 0.008, "min": 0.007, "max": 0.009},
        "n_batches": 1,
        "throughput_infers_per_s": 200.0,
        "speedup_vs_n1": 1.75,
        "parity_median": None,
    }
    csv_path = tmp_path / "probe_report.csv"

    probe._write_csv([row], csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1]
    assert data[header.index("speedup_vs_n1")] == "1.750000"


def _main_args(tmp_path: Path, n_models: str, mxq_path: Path) -> list[str]:
    """Build a ``sys.argv``-style list for driving ``probe.main`` end-to-end."""
    return [
        "probe_multi_model_parallel.py",
        "--mxq-path",
        str(mxq_path),
        "--n-models",
        n_models,
        "--dev-no",
        "0",
        "--core-mode",
        "single",
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--seq-len",
        "1",
        "--seed",
        "0",
        "--output-dir",
        str(tmp_path),
    ]


def test_main_warns_when_n1_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` prints a stderr warning when ``--n-models`` skips 1 so null speedups don't surprise callers."""
    _FakeModel._reset(launch_ok_before_fail=1_000_000)
    mxq_path = tmp_path / "fake.mxq"
    mxq_path.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", _main_args(tmp_path, "2,4", mxq_path))

    assert probe.main() == 0

    captured = capsys.readouterr()
    assert "--n-models does not include 1" in captured.err
    assert "speedup_vs_n1 will be reported" in captured.err
    # Console print for each N should show n/a rather than a fabricated value.
    assert "speedup_vs_n1=n/a" in captured.out


def test_main_no_warning_when_n1_included(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` does not emit the null-baseline warning when 1 is in the sweep."""
    _FakeModel._reset(launch_ok_before_fail=1_000_000)
    mxq_path = tmp_path / "fake.mxq"
    mxq_path.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", _main_args(tmp_path, "1,2", mxq_path))

    assert probe.main() == 0

    captured = capsys.readouterr()
    assert "does not include 1" not in captured.err


def test_main_reports_null_speedup_after_baseline_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the ``N=1`` run fails, subsequent ``N>1`` rows must report null speedup.

    ``main`` catches ``QbRuntimeError`` from ``_run_for_n(n=1, ...)`` and moves
    on to ``N=2, N=4``. The remaining rows must not fabricate a ``1.0``
    speedup — that was the pre-fix behavior that hid failed scaling.
    """
    _FakeModel._reset(launch_ok_before_fail=1_000_000)
    mxq_path = tmp_path / "fake.mxq"
    mxq_path.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", _main_args(tmp_path, "1,2,4", mxq_path))

    real_run_for_n = probe._run_for_n

    def _fail_only_n1(n: int, args: Any, baseline_per_model_median: Any) -> dict[str, Any]:
        if n == 1:
            raise _QBRUNTIME_ERROR("simulated N=1 baseline failure")
        return real_run_for_n(n, args, baseline_per_model_median)

    monkeypatch.setattr(probe, "_run_for_n", _fail_only_n1)

    assert probe.main() == 0

    captured = capsys.readouterr()
    assert "n_models=1: QbRuntimeError" in captured.err
    # After the baseline fails, N=2 and N=4 succeed but their speedups must be null.
    report_path = tmp_path / "probe_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["baseline_per_model_median_s"] is None
    assert "1" in report["errors"] or 1 in report["errors"]
    n_by_row = {row["n_models"]: row for row in report["rows"]}
    assert set(n_by_row.keys()) == {2, 4}
    assert n_by_row[2]["speedup_vs_n1"] is None
    assert n_by_row[4]["speedup_vs_n1"] is None
