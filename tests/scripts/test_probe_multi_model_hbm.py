"""Unit tests for the multi-model HBM saturation probe.

These tests exercise ``_sweep_device`` with a capacity-limited fake ``Model`` so
the schedule-endpoint-vs-attempted-count reporting can be verified without
touching real accelerators.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_PATH = _REPO_ROOT / "scripts" / "probe_multi_model_hbm.py"


def _load_probe_module() -> ModuleType:
    """Load ``scripts/probe_multi_model_hbm.py`` as an isolated module.

    The ``scripts/`` directory is not a Python package, so importing it through
    the normal machinery is not portable. A file-loader keeps the tests
    self-contained and avoids polluting ``sys.path``.
    """
    spec = importlib.util.spec_from_file_location("probe_multi_model_hbm_under_test", _PROBE_PATH)
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
    """Fake ``qbruntime.Model`` that fails after a caller-configurable capacity.

    Attributes:
        _created_ok: Number of ``_FakeModel`` instances successfully created since
            the last ``_reset``. Incremented on every successful ``__init__``.
        _capacity: Maximum number of successful ``__init__`` calls before the
            next one raises ``QbRuntimeError``.
    """

    _created_ok: int = 0
    _capacity: int = 0

    @classmethod
    def _reset(cls, capacity: int) -> None:
        cls._created_ok = 0
        cls._capacity = int(capacity)

    def __init__(self, mxq_path: str, model_config: Any) -> None:
        if _FakeModel._created_ok >= _FakeModel._capacity:
            raise _QBRUNTIME_ERROR("simulated BadAlloc")
        _FakeModel._created_ok += 1
        self.mxq_path = mxq_path

    def launch(self, acc: _FakeAccelerator) -> None:
        """No-op launch; allocation happens in ``__init__`` in this fake."""

    def dispose(self) -> None:
        """No-op dispose; the real runtime releases HBM here."""


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Monkey-patch the probe module to use hardware-independent fakes."""
    monkeypatch.setattr(probe, "Model", _FakeModel)
    monkeypatch.setattr(probe, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(probe, "_read_memory_mb", lambda dev_no, exe: (None, None))
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    return probe


def _args(schedule: str = "1,2,4,8", core_mode: str = "single") -> SimpleNamespace:
    """Build a minimal ``argparse``-shaped namespace for ``_sweep_device``."""
    return SimpleNamespace(
        n_schedule=schedule,
        n_models_max=32,
        core_mode=core_mode,
        mxq_path="/dev/null/fake.mxq",
        settle_s=0.0,
    )


def test_partial_step_failure_records_attempted_count(fake_probe: ModuleType) -> None:
    """Capacity between two schedule endpoints reports the actual failing model index."""
    _FakeModel._reset(capacity=6)
    report = fake_probe._sweep_device(dev_no=0, args=_args(), status_exe=None)

    # The 7th Model failed, not schedule endpoint 8.
    assert report["bad_alloc_at_n"] == 7
    # Six Models successfully launched (four at target_n=4 plus two more inside target_n=8).
    assert report["largest_ok_n"] == 6
    assert report["bad_alloc_error"] is not None

    failure_row = next(r for r in report["rows"] if not r["ok"])
    assert failure_row["n_models"] == 7
    assert failure_row["n_launched"] == 6

    # Successful rows still key on schedule endpoints (unchanged behavior).
    success_endpoints = [r["n_models"] for r in report["rows"] if r["ok"]]
    assert success_endpoints == [1, 2, 4]
    success_launched = [r["n_launched"] for r in report["rows"] if r["ok"]]
    assert success_launched == [1, 2, 4]


def test_capacity_at_schedule_endpoint_reports_next_index(fake_probe: ModuleType) -> None:
    """Capacity that exactly matches a schedule step reports the next attempted N."""
    _FakeModel._reset(capacity=4)
    report = fake_probe._sweep_device(dev_no=0, args=_args(), status_exe=None)

    # First failure occurs on the very next Model constructed after 4 succeeded.
    assert report["bad_alloc_at_n"] == 5
    assert report["largest_ok_n"] == 4

    failure_row = next(r for r in report["rows"] if not r["ok"])
    assert failure_row["n_models"] == 5
    assert failure_row["n_launched"] == 4


def test_capacity_exceeds_schedule_leaves_no_failure(fake_probe: ModuleType) -> None:
    """When the schedule fits inside capacity, no failure row is recorded."""
    _FakeModel._reset(capacity=16)
    report = fake_probe._sweep_device(dev_no=0, args=_args(), status_exe=None)

    assert report["bad_alloc_at_n"] is None
    assert report["bad_alloc_error"] is None
    # ``largest_ok_n`` follows the last successful step (schedule tail = 8).
    assert report["largest_ok_n"] == 8

    assert all(r["ok"] for r in report["rows"])
    assert [r["n_models"] for r in report["rows"]] == [1, 2, 4, 8]
    assert [r["n_launched"] for r in report["rows"]] == [1, 2, 4, 8]
