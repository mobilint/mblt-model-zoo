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
        self.disposed = False

    def launch(self, acc: _FakeAccelerator) -> None:
        """No-op launch; allocation happens in ``__init__`` in this fake."""

    def dispose(self) -> None:
        """Mark this fake handle as disposed so tests can assert release order."""
        self.disposed = True


class _LaunchFailModel:
    """Fake ``qbruntime.Model`` whose ``launch`` raises after a set of successes.

    ``__init__`` always succeeds (simulating LPDDR allocation during
    construction); ``launch`` succeeds ``launch_ok_before_fail`` times, then
    raises ``QbRuntimeError`` so ``_launch_up_to`` exercises the leak-fix path.
    """

    instances: list["_LaunchFailModel"] = []
    launch_ok_before_fail: int = 0
    _launched_ok: int = 0

    @classmethod
    def _reset(cls, launch_ok_before_fail: int) -> None:
        cls.instances = []
        cls.launch_ok_before_fail = int(launch_ok_before_fail)
        cls._launched_ok = 0

    def __init__(self, mxq_path: str, model_config: Any) -> None:
        self.mxq_path = mxq_path
        self.disposed = False
        _LaunchFailModel.instances.append(self)

    def launch(self, acc: _FakeAccelerator) -> None:
        if _LaunchFailModel._launched_ok >= _LaunchFailModel.launch_ok_before_fail:
            raise _QBRUNTIME_ERROR("simulated launch BadAlloc")
        _LaunchFailModel._launched_ok += 1

    def dispose(self) -> None:
        self.disposed = True


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


def test_launch_failure_disposes_unappended_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``mm.launch`` raises, ``_launch_up_to`` must dispose the orphan handle.

    Regression guard: without the fix, ``mm`` is constructed (allocating LPDDR)
    but never appended to ``models``, so the caller's ``_dispose_all`` cannot
    release it and subsequent probing rounds see the leaked handle as a
    cascading BadAlloc.
    """
    _LaunchFailModel._reset(launch_ok_before_fail=2)
    monkeypatch.setattr(probe, "Model", _LaunchFailModel)
    monkeypatch.setattr(probe, "Accelerator", _FakeAccelerator)

    models: list[Any] = []
    ok, err = probe._launch_up_to(
        acc=_FakeAccelerator(dev_no=0),
        models=models,
        target_n=4,
        mxq_path="/dev/null/fake.mxq",
        core_mode="single",
    )

    assert ok is False
    assert err is not None and "QbRuntimeError" in err
    # Two handles launched successfully and remain the caller's responsibility.
    assert len(models) == 2
    assert all(not m.disposed for m in models)
    # The third handle was constructed but its launch failed — it must be
    # disposed on the spot and NOT leak into ``models``.
    assert len(_LaunchFailModel.instances) == 3
    orphan = _LaunchFailModel.instances[-1]
    assert orphan not in models
    assert orphan.disposed is True


def test_construct_failure_does_not_touch_dispose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``Model(...)`` itself raises, there is no handle to dispose.

    ``_launch_up_to`` must surface the error unchanged and leave ``models`` at
    its previously successful length.
    """
    _FakeModel._reset(capacity=1)
    monkeypatch.setattr(probe, "Model", _FakeModel)
    monkeypatch.setattr(probe, "Accelerator", _FakeAccelerator)

    models: list[Any] = []
    ok, err = probe._launch_up_to(
        acc=_FakeAccelerator(dev_no=0),
        models=models,
        target_n=2,
        mxq_path="/dev/null/fake.mxq",
        core_mode="single",
    )

    assert ok is False
    assert err is not None and "QbRuntimeError" in err
    assert len(models) == 1
    assert models[0].disposed is False
