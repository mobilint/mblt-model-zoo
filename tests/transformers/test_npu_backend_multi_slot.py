"""Regression tests for :class:`MobilintNPUBackend` multi-slot lifecycle.

These tests stub out ``qbruntime.Accelerator`` / ``Model`` / ``ModelConfig`` so
we can exercise the multi-slot ``create`` / ``launch`` / rollback paths without
booting an NPU. They cover:

* ``N == 1``: single slot on a single device (the historical shape).
* ``N == 2``: two slots spread across two devices in round-robin order.
* ``BadAlloc`` mid-``create``: the failure surfaces as
  :class:`MobilintBackendAllocError` and every previously loaded slot is
  disposed and forgotten.
* ``BadAlloc`` mid-``launch``: same rollback semantics for the launch phase.
"""

from __future__ import annotations

from typing import List

import pytest
from qbruntime import QbRuntimeError

from mblt_model_zoo.utils import npu_backend as npu_backend_module
from mblt_model_zoo.utils.npu_backend import (
    MobilintBackendAllocError,
    MobilintNPUBackend,
)


class _FakeAccelerator:
    """Records the device number an accelerator was opened for."""

    def __init__(self, dev_no: int) -> None:
        self.dev_no = int(dev_no)


class _FakeModelConfig:
    """Records the core-mode selection so tests can assert per-slot config."""

    def __init__(self) -> None:
        self.mode: str | None = None
        self.cores: object | None = None
        self.clusters: object | None = None

    def set_single_core_mode(self, _batch_size: object, cores: object) -> None:
        self.mode = "single"
        self.cores = cores

    def set_multi_core_mode(self, clusters: object) -> None:
        self.mode = "multi"
        self.clusters = clusters

    def set_global4_core_mode(self, clusters: object) -> None:
        self.mode = "global4"
        self.clusters = clusters

    def set_global8_core_mode(self) -> None:
        self.mode = "global8"


class _FakeCacheInfo:
    """qbruntime CacheInfo stand-in exposing the fields the K probe reads."""

    def __init__(self, num_batches: int) -> None:
        self.num_batches = int(num_batches)


class _FakeModel:
    """qbruntime.Model stand-in that always succeeds and records lifecycle events."""

    def __init__(self, path: str, mc: _FakeModelConfig, k: int = 1, n_layers: int = 32) -> None:
        self.path = path
        self.mc = mc
        self.launched_on: _FakeAccelerator | None = None
        self.disposed = False
        self._cache_infos = [_FakeCacheInfo(k)] * int(n_layers)

    def get_cache_infos(self):
        return self._cache_infos

    def launch(self, acc: _FakeAccelerator) -> None:
        self.launched_on = acc

    def dispose(self) -> None:
        self.disposed = True


class _StubQbRuntime:
    """Container tracking every fake Model created via :func:`stub_qbruntime`."""

    def __init__(self) -> None:
        self.models: List[_FakeModel] = []
        self.create_should_fail_at: int | None = None
        self.launch_should_fail_at: int | None = None
        self.k_per_model: int = 1


@pytest.fixture
def stub_qbruntime(monkeypatch: pytest.MonkeyPatch) -> _StubQbRuntime:
    """Replace qbruntime symbols with fakes and return a tracker."""
    stub = _StubQbRuntime()

    def _model_factory(path: str, mc: _FakeModelConfig) -> _FakeModel:
        idx = len(stub.models)
        if stub.create_should_fail_at is not None and idx == stub.create_should_fail_at:
            raise QbRuntimeError(f"stub create failure at slot {idx}")
        model = _FakeModel(path, mc, k=stub.k_per_model)
        if stub.launch_should_fail_at is not None:
            slot_idx = idx

            def _failing_launch(acc: _FakeAccelerator, slot_idx: int = slot_idx) -> None:
                if slot_idx == stub.launch_should_fail_at:
                    raise QbRuntimeError(f"stub launch failure at slot {slot_idx}")
                model.launched_on = acc

            model.launch = _failing_launch  # type: ignore[assignment]
        stub.models.append(model)
        return model

    monkeypatch.setattr(npu_backend_module, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(npu_backend_module, "ModelConfig", _FakeModelConfig)
    monkeypatch.setattr(npu_backend_module, "Model", _model_factory)
    monkeypatch.setattr(npu_backend_module, "log_model_details", lambda *_a, **_k: None)
    return stub


def _make_backend_at(tmp_path, **kwargs) -> MobilintNPUBackend:
    """Build a backend whose ``mxq_path`` points at a stub file in ``tmp_path``."""
    mxq_path = tmp_path / "model.mxq"
    if not mxq_path.exists():
        mxq_path.write_bytes(b"stub")
    kwargs.setdefault("mxq_path", str(mxq_path))
    kwargs.setdefault("core_mode", "single")
    return MobilintNPUBackend(**kwargs)


def test_backend_create_n1_single_slot_single_device(tmp_path, stub_qbruntime) -> None:
    """max_batch_size=1 with a K=1 model yields exactly one slot on the target device."""
    backend = _make_backend_at(
        tmp_path,
        dev_no=0,
        max_batch_size=1,
        target_cores=["0:0:0", "0:0:1"],
    )

    backend.create()

    assert backend.n_models == 1
    assert backend.k_per_model == 1
    assert len(backend.mxq_models) == 1
    assert backend.model_dev_no == [0]
    assert list(backend.accs.keys()) == [0]
    assert backend.mxq_model is backend.mxq_models[0]  # compat shim


def test_backend_create_n2_round_robins_across_two_devices(tmp_path, stub_qbruntime) -> None:
    """max_batch_size=2 K=1 with two devices opens one slot per device in RR order."""
    backend = _make_backend_at(
        tmp_path,
        dev_no=[0, 1],
        max_batch_size=2,
        target_cores=["0:0:0", "1:0:0"],
    )

    backend.create()

    assert backend.n_models == 2
    assert backend.k_per_model == 1
    assert len(backend.mxq_models) == 2
    assert backend.model_dev_no == [0, 1]
    assert sorted(backend.accs.keys()) == [0, 1]


def test_backend_launch_forwards_each_slot_to_its_own_accelerator(tmp_path, stub_qbruntime) -> None:
    """launch() must hand every slot the accelerator opened for its assigned device."""
    backend = _make_backend_at(
        tmp_path,
        dev_no=[0, 1],
        max_batch_size=2,
        target_cores=["0:0:0", "1:0:0"],
    )
    backend.create()
    backend.launch()

    assert backend.mxq_models[0].launched_on is backend.accs[0]
    assert backend.mxq_models[1].launched_on is backend.accs[1]


def test_backend_create_badalloc_rolls_back_earlier_slots(tmp_path, stub_qbruntime) -> None:
    """A create-phase QbRuntimeError disposes every previously loaded slot."""
    stub_qbruntime.create_should_fail_at = 1  # slot 0 succeeds, slot 1 fails
    backend = _make_backend_at(
        tmp_path,
        dev_no=[0, 1],
        max_batch_size=2,
        target_cores=["0:0:0", "1:0:0"],
    )

    with pytest.raises(MobilintBackendAllocError) as excinfo:
        backend.create()

    err = excinfo.value
    assert err.phase == "create"
    assert err.slot == 1
    assert err.dev == 1
    assert err.succeeded_so_far == 1
    assert err.n_total >= 2
    assert err.max_batch_size == 2
    assert isinstance(err.original, QbRuntimeError)
    # Rollback drops every slot / accelerator handle so callers can retry.
    assert backend.mxq_models == []
    assert backend.accs == {}
    # The first slot must have been disposed on rollback.
    assert stub_qbruntime.models[0].disposed is True


def test_backend_launch_badalloc_rolls_back_earlier_slots(tmp_path, stub_qbruntime) -> None:
    """A launch-phase QbRuntimeError disposes every previously loaded slot."""
    stub_qbruntime.launch_should_fail_at = 1  # slot 0 launches, slot 1 fails
    backend = _make_backend_at(
        tmp_path,
        dev_no=[0, 1],
        max_batch_size=2,
        target_cores=["0:0:0", "1:0:0"],
    )
    backend.create()

    with pytest.raises(MobilintBackendAllocError) as excinfo:
        backend.launch()

    err = excinfo.value
    assert err.phase == "launch"
    assert err.slot == 1
    assert err.dev == 1
    assert err.succeeded_so_far == 1
    assert err.n_total == 2
    assert isinstance(err.original, QbRuntimeError)
    # Rollback disposes every slot including the one that already launched.
    assert backend.mxq_models == []
    assert backend.accs == {}
    assert all(m.disposed for m in stub_qbruntime.models)


def test_backend_dispose_is_idempotent(tmp_path, stub_qbruntime) -> None:
    """dispose() may be called multiple times without raising."""
    backend = _make_backend_at(
        tmp_path,
        dev_no=0,
        max_batch_size=1,
        target_cores=["0:0:0"],
    )
    backend.create()
    backend.dispose()
    backend.dispose()

    assert backend.mxq_models == []
    assert backend.accs == {}


def test_backend_probes_k_from_cache_infos_for_batched_llm(tmp_path, stub_qbruntime) -> None:
    """Batched-LLM MXQ (K=16) must be probed via get_cache_infos, not input shape.

    Regression: the previous input-shape-based probe read ``(1, -1, hidden)``
    for both batched and non-batched LLM MXQs and always returned K=1, so
    ``max_batch_size=16`` was expanded into 16 slots and hit BadAlloc.
    """
    stub_qbruntime.k_per_model = 16
    backend = _make_backend_at(
        tmp_path,
        dev_no=0,
        max_batch_size=16,
        target_cores=["0:0:0"],
    )

    backend.create()

    assert backend.k_per_model == 16
    assert backend.n_models == 1
    assert len(backend.mxq_models) == 1


def test_probe_k_per_model_falls_back_to_one_for_empty_cache_infos(tmp_path, stub_qbruntime) -> None:
    """Vision-style MXQs without KV cache layers must default to K=1."""

    class _NoCacheModel(_FakeModel):
        def get_cache_infos(self):
            return []

    k = MobilintNPUBackend._probe_k_per_model(_NoCacheModel("stub", _FakeModelConfig()))
    assert k == 1


def test_probe_k_per_model_falls_back_on_driver_error(tmp_path, stub_qbruntime) -> None:
    """A qbruntime error while probing must fall back to K=1 rather than propagate."""

    class _BrokenModel(_FakeModel):
        def get_cache_infos(self):
            raise QbRuntimeError("driver unhappy")

    k = MobilintNPUBackend._probe_k_per_model(_BrokenModel("stub", _FakeModelConfig()))
    assert k == 1


def test_backend_infer_slot_dispatches_to_the_selected_model(
    tmp_path, stub_qbruntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """infer_slot(i, x) must call ``.infer`` on ``mxq_models[i]``."""
    backend = _make_backend_at(
        tmp_path,
        dev_no=[0, 1],
        max_batch_size=2,
        target_cores=["0:0:0", "1:0:0"],
    )
    backend.create()

    calls: list[tuple[int, object]] = []
    for idx, model in enumerate(backend.mxq_models):
        monkeypatch.setattr(
            model,
            "infer",
            lambda x, _idx=idx: (calls.append((_idx, x)), x)[-1],
            raising=False,
        )

    backend.infer_slot(0, "payload-0")
    backend.infer_slot(1, "payload-1")

    assert calls == [(0, "payload-0"), (1, "payload-1")]
