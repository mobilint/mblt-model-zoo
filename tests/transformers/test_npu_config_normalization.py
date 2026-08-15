"""Unit tests for NPU target-field normalization and atomic ``NPUTargetSpec`` replace.

Covers legacy migration, ``dev_no`` sugar expansion, grain fold/unfold,
``global8`` coverage validation, canonical round-trip via
``MobilintNPUBackend.to_dict`` / ``from_dict``, and the atomic-replace
setter contract that eliminates HF ``from_pretrained``'s per-field
setattr race.
"""

from __future__ import annotations

import warnings

import pytest

from mblt_model_zoo.hf_transformers.utils.configuration_utils import (
    _migrate_target_clusters,
    _migrate_target_cores,
    _normalize_npu_target_kwargs,
)
from mblt_model_zoo.utils.npu_backend import MobilintNPUBackend

# ---------------------------------------------------------------------------
# Legacy migration heuristic (per-item colon count)
# ---------------------------------------------------------------------------


def test_legacy_two_part_target_cores_gets_dev_no_prefix() -> None:
    """A legacy ``"c:k"`` core is migrated to ``"d:c:k"`` using ``dev_no``."""
    kwargs = {"core_mode": "single", "dev_no": 1, "target_cores": ["0:0"]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["1:0:0"]


def test_canonical_three_part_target_cores_passes_through() -> None:
    """A fully-qualified ``"d:c:k"`` core is preserved verbatim."""
    kwargs = {"core_mode": "single", "target_cores": ["0:0:0"]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["0:0:0"]


def test_mixed_legacy_and_canonical_target_cores_raises() -> None:
    """Mixing 2-part legacy and 3-part canonical items in one list is rejected."""
    kwargs = {"core_mode": "single", "target_cores": ["0:0", "1:0:0"]}
    with pytest.raises(ValueError, match="mixes legacy"):
        _normalize_npu_target_kwargs(kwargs)


def test_legacy_target_cores_with_list_dev_no_raises() -> None:
    """A legacy target item is ambiguous when ``dev_no`` is a list."""
    kwargs = {"core_mode": "single", "dev_no": [0, 1], "target_cores": ["0:0"]}
    with pytest.raises(ValueError, match="ambiguous"):
        _normalize_npu_target_kwargs(kwargs)


def test_legacy_bare_int_target_clusters_gets_dev_no_prefix() -> None:
    """A legacy bare int cluster is migrated to canonical ``"d:c"``."""
    kwargs = {"core_mode": "global4", "dev_no": 1, "target_clusters": [0]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["1:0"]


def test_canonical_two_part_target_clusters_passes_through() -> None:
    """A canonical ``"d:c"`` cluster is preserved verbatim."""
    kwargs = {"core_mode": "global4", "target_clusters": ["0:0"]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0"]


def test_mixed_target_clusters_raises() -> None:
    """Mixing an int (legacy) and a canonical ``"d:c"`` string is rejected."""
    kwargs = {"core_mode": "global4", "target_clusters": [0, "1:0"]}
    with pytest.raises(ValueError, match="mixes legacy"):
        _normalize_npu_target_kwargs(kwargs)


def test_legacy_target_clusters_with_list_dev_no_raises() -> None:
    """A legacy bare int cluster is ambiguous when ``dev_no`` is a list."""
    kwargs = {"core_mode": "global4", "dev_no": [0, 1], "target_clusters": [0]}
    with pytest.raises(ValueError, match="ambiguous"):
        _normalize_npu_target_kwargs(kwargs)


# ---------------------------------------------------------------------------
# dev_no sugar expansion (both target lists absent)
# ---------------------------------------------------------------------------


def test_scalar_dev_no_single_mode_expands_to_full_device() -> None:
    """``dev_no=0`` with no targets under ``single`` mode fills all 8 cores."""
    kwargs = {"core_mode": "single", "dev_no": 0}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == [
        "0:0:0",
        "0:0:1",
        "0:0:2",
        "0:0:3",
        "0:1:0",
        "0:1:1",
        "0:1:2",
        "0:1:3",
    ]
    assert "target_clusters" not in kwargs


def test_list_dev_no_global4_mode_expands_to_both_clusters_per_device() -> None:
    """``dev_no=[0, 1]`` with no targets under ``global4`` fills 4 clusters total."""
    kwargs = {"core_mode": "global4", "dev_no": [0, 1]}
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0", "0:1", "1:0", "1:1"]
    assert "target_cores" not in kwargs


# ---------------------------------------------------------------------------
# Grain fold / unfold
# ---------------------------------------------------------------------------


def test_single_mode_expands_target_clusters_to_target_cores_without_warning() -> None:
    """``single`` mode with only ``target_clusters`` unfolds to every core, no warn."""
    kwargs = {"core_mode": "single", "target_clusters": ["0:0"]}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["0:0:0", "0:0:1", "0:0:2", "0:0:3"]
    assert not [w for w in recorded if issubclass(w.category, UserWarning)]


def test_global4_mode_folds_full_target_cores_to_clusters_without_warning() -> None:
    """``global4`` mode with a complete 4-core cluster folds without warning."""
    kwargs = {
        "core_mode": "global4",
        "target_cores": ["0:0:0", "0:0:1", "0:0:2", "0:0:3"],
    }
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0"]
    assert not [w for w in recorded if issubclass(w.category, UserWarning)]


def test_global4_mode_folds_incomplete_target_cores_and_warns() -> None:
    """``global4`` mode rounds a partial cluster up to whole with ``UserWarning``."""
    kwargs = {"core_mode": "global4", "target_cores": ["0:0:0", "0:0:1"]}
    with pytest.warns(UserWarning, match="rounded up"):
        _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0"]


# ---------------------------------------------------------------------------
# global8 coverage
# ---------------------------------------------------------------------------


def test_global8_missing_cluster_1_on_device_raises() -> None:
    """``global8`` requires both clusters on every device; missing one raises."""
    kwargs = {
        "core_mode": "global8",
        "target_cores": [
            "0:0:0",
            "0:0:1",
            "0:0:2",
            "0:0:3",
        ],
    }
    with pytest.raises(ValueError, match="global8"):
        _normalize_npu_target_kwargs(kwargs)


def test_global8_covers_both_clusters_on_each_device() -> None:
    """``global8`` passes when every device carries both cluster indices."""
    kwargs = {
        "core_mode": "global8",
        "dev_no": [0, 1],
        "target_clusters": ["0:0", "0:1", "1:0", "1:1"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0", "0:1", "1:0", "1:1"]


# ---------------------------------------------------------------------------
# Device-set consistency
# ---------------------------------------------------------------------------


def test_explicit_dev_no_must_match_target_device_set() -> None:
    """When ``dev_no`` is provided, targets must reference exactly its device set."""
    kwargs = {"core_mode": "single", "dev_no": 0, "target_cores": ["1:0:0"]}
    with pytest.raises(ValueError, match="does not match"):
        _normalize_npu_target_kwargs(kwargs)


# ---------------------------------------------------------------------------
# Stale off-mode grain is dropped (PR #109 review P2 finding [4])
# ---------------------------------------------------------------------------


def test_single_mode_drops_stale_target_clusters() -> None:
    """``single`` mode drops a ``target_clusters`` entry that would otherwise leak into device set."""
    kwargs = {
        "core_mode": "single",
        "target_cores": ["0:0:0"],
        "target_clusters": ["1:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["0:0:0"]
    assert "target_clusters" not in kwargs


def test_single_mode_stale_target_clusters_does_not_break_dev_no_consistency() -> None:
    """Stale ``target_clusters`` on a different device must not trigger the dev_no mismatch check."""
    kwargs = {
        "core_mode": "single",
        "dev_no": 0,
        "target_cores": ["0:0:0"],
        "target_clusters": ["1:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_cores"] == ["0:0:0"]
    assert "target_clusters" not in kwargs


def test_global4_mode_drops_stale_target_cores() -> None:
    """A non-``single`` mode drops a stale ``target_cores`` entry that references a foreign device."""
    kwargs = {
        "core_mode": "global4",
        "target_cores": ["1:0:0"],
        "target_clusters": ["0:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0"]
    assert "target_cores" not in kwargs


def test_global4_mode_stale_target_cores_does_not_break_dev_no_consistency() -> None:
    """Stale ``target_cores`` on a foreign device must not trigger the dev_no mismatch check."""
    kwargs = {
        "core_mode": "global4",
        "dev_no": 0,
        "target_cores": ["1:0:0"],
        "target_clusters": ["0:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    assert kwargs["target_clusters"] == ["0:0"]
    assert "target_cores" not in kwargs


# ---------------------------------------------------------------------------
# Backend round-trip
# ---------------------------------------------------------------------------


def test_backend_from_dict_emits_new_form_after_normalization() -> None:
    """After config-level normalization the backend serializes canonical strings."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": 0,
        "target_cores": ["0:0"],  # legacy 2-part input
    }
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = backend.to_dict()
    assert dumped["target_cores"] == ["0:0:0"]
    assert dumped["dev_no"] == 0
    assert dumped["core_mode"] == "single"


def test_backend_roundtrip_preserves_canonical_target_cores() -> None:
    """A canonical config round-trips through ``to_dict`` / ``from_dict`` unchanged."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": 0,
        "target_cores": ["0:0:0", "0:0:1"],
    }
    _normalize_npu_target_kwargs(kwargs)
    first = MobilintNPUBackend.from_dict(dict(kwargs))
    round_trip_kwargs = first.to_dict()
    _normalize_npu_target_kwargs(round_trip_kwargs)
    second = MobilintNPUBackend.from_dict(dict(round_trip_kwargs))
    assert second.to_dict()["target_cores"] == ["0:0:0", "0:0:1"]
    assert second.to_dict()["dev_no"] == 0


def test_backend_roundtrip_preserves_list_dev_no() -> None:
    """A list-shaped ``dev_no`` survives the round-trip without collapsing."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "global4",
        "dev_no": [0, 1],
    }
    _normalize_npu_target_kwargs(kwargs)
    first = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = first.to_dict()
    assert dumped["dev_no"] == [0, 1]
    assert dumped["target_clusters"] == ["0:0", "0:1", "1:0", "1:1"]


def test_backend_roundtrip_preserves_global8_coverage() -> None:
    """A ``global8`` config with dual-cluster coverage round-trips correctly."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "global8",
        "target_clusters": [0, 1],
    }
    _normalize_npu_target_kwargs(kwargs)
    first = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = first.to_dict()
    _normalize_npu_target_kwargs(dumped)
    second = MobilintNPUBackend.from_dict(dict(dumped))
    assert second.to_dict()["target_clusters"] == ["0:0", "0:1"]


def test_backend_roundtrip_multi_device_target_without_explicit_dev_no() -> None:
    """Canonical multi-device targets survive round-trip without user-supplied ``dev_no``.

    Reproduces the P2 finding on PR #109: a user pins two devices via
    ``target_cores`` alone, ``to_dict`` used to emit the backend default
    ``dev_no=0``, and the next ``_normalize_npu_target_kwargs`` rejected
    the pair as a device-set mismatch.
    """
    kwargs = {"core_mode": "single", "target_cores": ["0:0:0", "1:0:0"]}
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = backend.to_dict()
    assert dumped["target_cores"] == ["0:0:0", "1:0:0"]
    assert dumped["dev_no"] == [0, 1]
    _normalize_npu_target_kwargs(dumped)
    reloaded = MobilintNPUBackend.from_dict(dict(dumped))
    assert reloaded.to_dict()["target_cores"] == ["0:0:0", "1:0:0"]
    assert reloaded.to_dict()["dev_no"] == [0, 1]


def test_backend_roundtrip_single_device_dev_no_stays_scalar() -> None:
    """A single-device explicit ``dev_no=0`` keeps its scalar shape on round-trip."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": 0,
        "target_cores": ["0:0:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = backend.to_dict()
    assert dumped["dev_no"] == 0
    assert dumped["target_cores"] == ["0:0:0"]
    _normalize_npu_target_kwargs(dumped)
    reloaded = MobilintNPUBackend.from_dict(dict(dumped))
    assert reloaded.to_dict()["dev_no"] == 0
    assert reloaded.to_dict()["target_cores"] == ["0:0:0"]


def test_backend_roundtrip_multi_device_dev_no_stays_list() -> None:
    """Explicit ``dev_no=[0, 1]`` with matching multi-device target survives round-trip."""
    kwargs = {
        "mxq_path": "model.mxq",
        "core_mode": "single",
        "dev_no": [0, 1],
        "target_cores": ["0:0:0", "1:0:0"],
    }
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    dumped = backend.to_dict()
    assert dumped["dev_no"] == [0, 1]
    assert dumped["target_cores"] == ["0:0:0", "1:0:0"]
    _normalize_npu_target_kwargs(dumped)
    reloaded = MobilintNPUBackend.from_dict(dict(dumped))
    assert reloaded.to_dict()["dev_no"] == [0, 1]
    assert reloaded.to_dict()["target_cores"] == ["0:0:0", "1:0:0"]


# ---------------------------------------------------------------------------
# Atomic-replace setter contract (``NPUTargetSpec._with``)
#
# HF ``from_pretrained`` applies CLI ``model_kwargs`` via ``setattr`` after the
# config layer already built the initial :class:`NPUTargetSpec` from the JSON
# payload. Each per-field setter atomically replaces the backend's
# ``_spec`` through :meth:`NPUTargetSpec._with`, so the four target-topology
# fields never disagree — there is no partial-state moment between setattrs
# that a snapshot-diff reconciliation pass would have to detect and fix.
# ---------------------------------------------------------------------------


def _load_backend(kwargs: dict) -> MobilintNPUBackend:
    """Normalize ``kwargs`` and load a backend at a clean canonical baseline."""
    _normalize_npu_target_kwargs(kwargs)
    return MobilintNPUBackend.from_dict(dict(kwargs))


def test_atomic_replace_scalar_dev_no_override_rebuilds_targets_to_new_device() -> None:
    """``dev_no`` override to device 1 clears device-0 targets and re-expands under ``single``."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    # HF ``setattr`` fires atomically: the setter replaces ``backend._spec``
    # via :meth:`NPUTargetSpec._with`, which clears the stale JSON targets
    # and re-expands the ``dev_no`` sugar for the new device.
    backend.dev_no = 1
    assert backend._target_cores_serialized == [
        "1:0:0",
        "1:0:1",
        "1:0:2",
        "1:0:3",
        "1:1:0",
        "1:1:1",
        "1:1:2",
        "1:1:3",
    ]
    assert backend._target_clusters_serialized == []
    assert backend.dev_no == 1


def test_atomic_replace_list_dev_no_override_rebuilds_targets_across_devices() -> None:
    """``dev_no=[0, 1]`` override with stale single-device targets re-expands to both devices."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    backend.dev_no = [0, 1]
    # Sugar expansion under single mode fills all 8 cores on each device.
    assert backend._target_cores_serialized == [f"{d}:{c}:{k}" for d in (0, 1) for c in (0, 1) for k in range(4)]
    assert backend._target_clusters_serialized == []
    assert backend.dev_no == [0, 1]


def test_atomic_replace_consistent_state_is_noop() -> None:
    """Setting a field to its current canonical value leaves the spec unchanged."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0", "0:0:1"],
        }
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        backend.core_mode = "single"  # no-op setter
    assert backend._target_cores_serialized == ["0:0:0", "0:0:1"]
    assert not [w for w in recorded if issubclass(w.category, UserWarning)]


def test_atomic_replace_core_mode_change_folds_cores_to_clusters() -> None:
    """Switching ``core_mode`` to ``global4`` folds a full-cluster core list to a cluster string."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0", "0:0:1", "0:0:2", "0:0:3"],
        }
    )
    backend.core_mode = "global4"
    assert backend._target_clusters_serialized == ["0:0"]
    # ``NPUTargetSpec.from_kwargs`` drops the off-mode grain under global4.
    assert backend._target_cores_serialized == []


def test_atomic_replace_p2_core_mode_and_cluster_override_syncs_dev_no() -> None:
    """PR #109 P2: ``core_mode=global4`` + ``target_clusters=["1:0"]`` overrides sync dev_no atomically.

    Reproduces the failing path: a JSON config with ``core_mode=single`` +
    dev0 target_cores. The caller overrides ``core_mode`` to ``global4``
    and ``target_clusters`` to a device-1 cluster without redoing
    ``dev_no``. Under the old snapshot-diff reconciliation the two
    overrides were reconciled together, unioning stale dev0 cores with
    new dev1 clusters into ``dev_no=[0, 1]``, then dropping the stale
    off-mode grain reduced the target device set to ``{1}`` and the
    device-set consistency check raised. Under the atomic-replace
    contract each setter fires independently: the ``core_mode`` override
    folds the dev0 cores to a dev0 cluster, then the ``target_clusters``
    override synchronously replaces both the clusters and ``dev_no`` in
    a single atomic ``NPUTargetSpec._with`` call.
    """
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": [f"0:{c}:{k}" for c in (0, 1) for k in range(4)],
        }
    )
    # Simulate HF ``model_kwargs`` application: two independent setattrs.
    backend.core_mode = "global4"
    backend.target_clusters = ["1:0"]

    assert backend.dev_no == 1
    assert backend.core_mode == "global4"
    assert backend._target_clusters_serialized == ["1:0"]
    assert backend._target_cores_serialized == []


def test_atomic_replace_target_clusters_before_core_mode_matches_reverse_order() -> None:
    """PR #109 review: ``target_clusters`` BEFORE ``core_mode`` must converge on the same spec.

    HF ``from_pretrained`` calls ``setattr`` on each ``model_kwargs`` key in
    whatever order the dict yields them, so ``target_clusters`` may land
    before ``core_mode`` on an override that switches both. Under the old
    ``_with`` logic the target-only path unioned the stale dev0 cores with
    the new ``"1:0"`` cluster into ``dev_no=[0, 1]``, and then
    :meth:`NPUTargetSpec.from_kwargs` dropped the off-mode ``target_clusters``
    (mode is still ``single`` at that moment) and raised a device-set
    mismatch. The sibling-clear in :meth:`_with` makes the setter order
    symmetric.
    """
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": [f"0:{c}:{k}" for c in (0, 1) for k in range(4)],
        }
    )
    backend.target_clusters = ["1:0"]
    backend.core_mode = "global4"

    assert backend.dev_no == 1
    assert backend.core_mode == "global4"
    assert backend._target_clusters_serialized == ["1:0"]
    assert backend._target_cores_serialized == []


def test_atomic_replace_target_cores_before_core_mode_matches_reverse_order() -> None:
    """Mirror: full-cluster ``target_cores`` BEFORE ``core_mode='global4'`` from a single-mode config.

    Starts from the same single-mode config with pre-populated dev0 cores
    as the previous test, but overrides via the ``target_cores`` grain
    instead. A full-cluster core list (``1:0:{0..3}``) avoids the lossy
    partial-cluster fold, so both setter orders must converge on the same
    ``NPUTargetSpec``. Without the sibling-clear the ``target_cores``
    setter would carry the stale dev0 cores through and the target-only
    ``dev_no`` sync would still work here (both grains resolve to the
    same device set), but exercising this order guards against future
    regressions in the mirror path.
    """
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": [f"0:{c}:{k}" for c in (0, 1) for k in range(4)],
        }
    )
    backend.target_cores = [f"1:0:{k}" for k in range(4)]
    backend.core_mode = "global4"

    assert backend.dev_no == 1
    assert backend.core_mode == "global4"
    assert backend._target_clusters_serialized == ["1:0"]
    assert backend._target_cores_serialized == []


@pytest.mark.parametrize(
    "target_key,target_value",
    [
        ("target_clusters", ["1:0"]),
        ("target_cores", [f"1:0:{k}" for k in range(4)]),
    ],
)
def test_atomic_replace_setter_order_is_symmetric_across_permutations(target_key: str, target_value: list) -> None:
    """Both HF ``setattr`` orders (core_mode-then-target, target-then-core_mode) yield the same spec.

    Runs the two-setter permutation for both target grains starting from
    the same single-mode config with pre-populated dev0 cores, asserting
    the resulting :class:`NPUTargetSpec` is identical. This is the
    parity check the PR #109 review asked for.
    """

    def _apply(order: tuple[str, ...]) -> tuple:
        backend = _load_backend(
            {
                "mxq_path": "model.mxq",
                "core_mode": "single",
                "dev_no": 0,
                "target_cores": [f"0:{c}:{k}" for c in (0, 1) for k in range(4)],
            }
        )
        for key in order:
            if key == "core_mode":
                backend.core_mode = "global4"
            else:
                setattr(backend, target_key, target_value)
        return (
            backend.dev_no,
            backend.core_mode,
            tuple(backend._target_cores_serialized),
            tuple(backend._target_clusters_serialized),
        )

    forward = _apply(("core_mode", target_key))
    reverse = _apply((target_key, "core_mode"))
    assert forward == reverse


def test_atomic_replace_default_bare_dev_no_leaves_expanded_targets_intact() -> None:
    """Bare ``dev_no=0`` default with sugar-expanded targets stays canonical."""
    kwargs = {"mxq_path": "model.mxq", "core_mode": "single", "dev_no": 0}
    _normalize_npu_target_kwargs(kwargs)
    backend = MobilintNPUBackend.from_dict(dict(kwargs))
    assert backend._target_cores_serialized == [f"0:{c}:{k}" for c in (0, 1) for k in range(4)]


# ---------------------------------------------------------------------------
# Codex review regressions: legacy CoreId / Cluster ambiguity, target
# authority, CLI-layer legacy defaults with list dev_no.
# ---------------------------------------------------------------------------


def test_legacy_coreid_with_list_dev_no_raises_symmetrically() -> None:
    """``CoreId`` objects lack a device prefix; reject them with a list-valued ``dev_no``."""
    from qbruntime import Cluster, Core, CoreId

    with pytest.raises(ValueError, match="Legacy CoreId"):
        _migrate_target_cores(
            [CoreId(Cluster.Cluster0, Core.Core0)],
            fallback_dev=0,
            dev_no_is_list=True,
        )


def test_legacy_cluster_object_with_list_dev_no_raises_symmetrically() -> None:
    """``Cluster`` objects lack a device prefix; reject them with a list-valued ``dev_no``."""
    from qbruntime import Cluster

    with pytest.raises(ValueError, match="Legacy Cluster"):
        _migrate_target_clusters(
            [Cluster.Cluster0],
            fallback_dev=0,
            dev_no_is_list=True,
        )


def test_legacy_coreid_with_scalar_dev_no_produces_canonical_form() -> None:
    """Scalar ``dev_no`` prefixes ``CoreId`` objects; canonical output uses integer indices."""
    from qbruntime import Cluster, Core, CoreId

    result = _migrate_target_cores(
        [CoreId(Cluster.Cluster0, Core.Core3), CoreId(Cluster.Cluster1, Core.Core0)],
        fallback_dev=1,
        dev_no_is_list=False,
    )
    assert result == ["1:0:3", "1:1:0"]


def test_legacy_cluster_object_with_scalar_dev_no_produces_canonical_form() -> None:
    """Scalar ``dev_no`` prefixes ``Cluster`` objects; canonical output uses integer indices."""
    from qbruntime import Cluster

    result = _migrate_target_clusters(
        [Cluster.Cluster0, Cluster.Cluster1],
        fallback_dev=1,
        dev_no_is_list=False,
    )
    assert result == ["1:0", "1:1"]


def test_atomic_replace_target_override_syncs_dev_no_and_preserves_targets() -> None:
    """Target override without a matching ``dev_no`` treats the canonical target as authoritative."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    # Simulate ``--vision-target-cores 1:0:0``: target setter fires alone,
    # the caller left ``dev_no`` at its default. The canonical target
    # unambiguously specifies device 1; ``dev_no`` syncs to match instead
    # of clobbering the caller's target.
    backend.target_cores = ["1:0:0"]
    assert backend.dev_no == 1
    assert backend._target_cores_serialized == ["1:0:0"]


def test_atomic_replace_target_override_across_multi_device_sets_dev_no_list() -> None:
    """A canonical multi-device target override syncs ``dev_no`` to the target list."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    backend.target_cores = ["0:0:0", "1:0:0"]
    assert backend.dev_no == [0, 1]
    assert backend._target_cores_serialized == ["0:0:0", "1:0:0"]


def test_atomic_replace_dev_no_and_target_both_overridden_consistently_passes() -> None:
    """When ``dev_no`` and targets are both overridden consistently, both take effect."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    backend.dev_no = 1
    backend.target_cores = ["1:0:0"]
    assert backend.dev_no == 1
    assert backend._target_cores_serialized == ["1:0:0"]


def test_atomic_replace_dev_no_and_target_both_overridden_inconsistently_raises() -> None:
    """When both are overridden but disagree, the consistency check raises on the second setter."""
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    # The first setter clears stale targets and re-expands under the new dev_no.
    backend.dev_no = 1
    # The second setter marks targets as caller-overridden while dev_no is
    # still marked overridden from the previous call; the atomic replace
    # runs the device-set consistency check and rejects the mismatch.
    with pytest.raises(ValueError, match="target device set"):
        backend.target_cores = ["2:0:0"]


def test_atomic_replace_prefix_scoped_kwargs_work_independently() -> None:
    """``vision_`` and ``text_`` prefix keys are normalized independently."""
    kwargs = {
        "vision_dev_no": 0,
        "vision_core_mode": "single",
        "vision_target_cores": ["0:0"],  # legacy 2-part
        "text_dev_no": 1,
        "text_core_mode": "global4",
        "text_target_clusters": [0],  # legacy bare int
    }
    _normalize_npu_target_kwargs(kwargs, prefix="vision_")
    _normalize_npu_target_kwargs(kwargs, prefix="text_")
    assert kwargs["vision_target_cores"] == ["0:0:0"]
    assert kwargs["text_target_clusters"] == ["1:0"]


def test_cli_apply_core_mode_suppresses_single_default_for_list_dev_no() -> None:
    """CLI helper suppresses the ``"0:0"`` single-mode default when ``dev_no`` is a list."""
    from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import apply_core_mode_model_kwargs

    model_kwargs: dict = {}
    apply_core_mode_model_kwargs(
        model_kwargs,
        "single",
        dev_no=[0, 1],
        default_single_target_cores=("0:0",),
    )
    assert "target_cores" not in model_kwargs
    assert model_kwargs["core_mode"] == "single"


def test_cli_apply_core_mode_suppresses_global4_default_for_list_dev_no() -> None:
    """CLI helper suppresses the ``[0]`` global4 cluster default when ``dev_no`` is a list."""
    from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import apply_core_mode_model_kwargs

    model_kwargs: dict = {}
    apply_core_mode_model_kwargs(model_kwargs, "global4", dev_no=[0, 1])
    assert "target_clusters" not in model_kwargs
    assert model_kwargs["core_mode"] == "global4"


def test_cli_apply_core_mode_suppresses_global8_default_for_list_dev_no() -> None:
    """CLI helper suppresses the ``[0, 1]`` global8 cluster default when ``dev_no`` is a list."""
    from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import apply_core_mode_model_kwargs

    model_kwargs: dict = {}
    apply_core_mode_model_kwargs(model_kwargs, "global8", dev_no=[0, 1])
    assert "target_clusters" not in model_kwargs
    assert model_kwargs["core_mode"] == "global8"


def test_cli_apply_core_mode_still_injects_defaults_for_scalar_dev_no() -> None:
    """Scalar ``dev_no`` (or ``None``) keeps the legacy defaults so single-device UX is unchanged."""
    from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import apply_core_mode_model_kwargs

    for scalar_dev_no in (None, 0, 1):
        for core_mode, default_key, default_value in (
            ("single", "target_cores", ["0:0"]),
            ("global4", "target_clusters", [0]),
            ("global8", "target_clusters", [0, 1]),
        ):
            model_kwargs: dict = {}
            apply_core_mode_model_kwargs(
                model_kwargs,
                core_mode,
                dev_no=scalar_dev_no,
                default_single_target_cores=("0:0",),
            )
            assert model_kwargs.get(default_key) == default_value, (
                f"expected {default_key}={default_value} for scalar dev_no={scalar_dev_no}, "
                f"mode={core_mode}; got {model_kwargs}"
            )


def test_cli_apply_subconfig_threads_dev_no_to_each_prefix() -> None:
    """VLM/EAGLE-3 subconfig helper honors per-prefix ``dev_no`` for default suppression."""
    from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
        apply_subconfig_core_mode_model_kwargs,
    )

    model_kwargs: dict = {}
    apply_subconfig_core_mode_model_kwargs(
        model_kwargs,
        ("vision", "text"),
        "single",
        base_dev_no=0,
        subconfig_dev_nos={"vision": 0, "text": [0, 1]},
        default_single_target_cores=("0:0",),
    )
    # vision (scalar dev_no) keeps the default; text (list dev_no) suppresses it.
    assert model_kwargs.get("vision_target_cores") == ["0:0"]
    assert "text_target_cores" not in model_kwargs


# ---------------------------------------------------------------------------
# PR #109 P2 (second surface): NPUTargetSpec.from_kwargs must derive dev_no
# from canonical targets when the caller does not pin one, so the in-memory
# spec is self-consistent for a subsequent :meth:`_with` override.
# ---------------------------------------------------------------------------


def test_from_kwargs_derives_scalar_dev_no_from_target_cores() -> None:
    """Canonical single-device ``target_cores`` without ``dev_no`` pins ``spec.dev_no`` to that device."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"target_cores": ["1:0:0"]})
    assert spec.dev_no == 1
    assert spec.cores == ("1:0:0",)


def test_from_kwargs_derived_dev_no_survives_subsequent_core_mode_with() -> None:
    """PR #109 P2 reproducer: `_with(core_mode="global4")` after canonical-target load must not raise."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"target_cores": ["1:0:0"]})
    # Under the old behavior, ``spec.dev_no`` was ``0`` even though targets
    # named device 1; the ``_with`` call then passed ``dev_no=0`` back through
    # ``from_kwargs`` with ``dev_no_given=True`` and the device-set consistency
    # check raised. The derivation makes the in-memory spec self-consistent.
    updated = spec._with(core_mode="global4")
    assert updated.dev_no == 1
    assert updated.clusters == ("1:0",)
    assert updated.cores == ()


def test_from_kwargs_derives_scalar_dev_no_from_target_clusters() -> None:
    """Canonical single-device ``target_clusters`` without ``dev_no`` pins ``spec.dev_no`` to that device."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"core_mode": "global4", "target_clusters": ["2:0", "2:1"]})
    assert spec.dev_no == 2


def test_from_kwargs_derives_sorted_tuple_dev_no_from_multi_device_targets() -> None:
    """Canonical multi-device ``target_cores`` without ``dev_no`` derives a sorted tuple."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"target_cores": ["1:0:0", "2:0:0"], "core_mode": "single"})
    assert spec.dev_no == (1, 2)


def test_from_kwargs_explicit_dev_no_still_authoritative_and_raises_on_mismatch() -> None:
    """An explicit ``dev_no`` still enforces the device-set consistency check."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    with pytest.raises(ValueError, match="does not match"):
        NPUTargetSpec.from_kwargs({"target_cores": ["1:0:0"], "dev_no": 0})


def test_from_kwargs_legacy_two_part_cores_still_respect_explicit_dev_no() -> None:
    """Explicit ``dev_no`` is preserved when legacy 2-part cores are migrated."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"target_cores": ["0:0"], "dev_no": 5})
    assert spec.dev_no == 5
    assert spec.cores == ("5:0:0",)


def test_from_kwargs_prefix_scoped_targets_derive_dev_no() -> None:
    """Prefixed canonical targets without a matching prefixed ``dev_no`` derive ``spec.dev_no``."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"vision_target_cores": ["3:0:0"]}, prefix="vision_")
    assert spec.dev_no == 3
    assert spec.cores == ("3:0:0",)


# ---------------------------------------------------------------------------
# PR #109 P2 (third surface): NPUTargetSpec._with target-only override must
# migrate legacy items (CoreId/Cluster, "c:k", bare ints) to canonical form
# before deriving the device set. Otherwise ``_devices_from_targets`` reads
# ``split(":", 1)[0]`` on non-canonical inputs and either raises
# ``AttributeError`` (CoreId/Cluster objects) or silently mis-reads the
# cluster component as the device index (legacy 2-part strings).
# ---------------------------------------------------------------------------


def test_with_target_only_override_accepts_coreid_object_on_scalar_dev_no() -> None:
    """A legacy ``CoreId`` target override no longer raises ``AttributeError``."""
    from qbruntime import Cluster, Core, CoreId

    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"dev_no": 3, "core_mode": "single", "target_cores": ["3:0:0"]})
    updated = spec._with(target_cores=[CoreId(Cluster.Cluster0, Core.Core1)])
    assert updated.dev_no == 3
    assert updated.cores == ("3:0:1",)


def test_with_target_only_override_migrates_legacy_two_part_using_inherited_dev_no() -> None:
    """A legacy ``"c:k"`` target override adopts ``self.dev_no`` as the device prefix, not ``0``."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"dev_no": 3, "core_mode": "single", "target_cores": ["3:0:0"]})
    updated = spec._with(target_cores=["0:1"])
    # Under the pre-fix code, ``_devices_from_targets`` read the ``"0"`` prefix
    # as the device index, synced ``dev_no`` to ``0``, and :meth:`from_kwargs`
    # then migrated ``"0:1"`` with fallback ``0`` — producing ``"0:0:1"`` and
    # silently dropping the intended device 3.
    assert updated.dev_no == 3
    assert updated.cores == ("3:0:1",)


def test_with_target_only_override_canonical_string_still_syncs_dev_no() -> None:
    """Fully-qualified canonical target overrides still sync ``dev_no`` to the target device."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"dev_no": 3, "core_mode": "single", "target_cores": ["3:0:0"]})
    updated = spec._with(target_cores=["5:0:1"])
    assert updated.dev_no == 5
    assert updated.cores == ("5:0:1",)


def test_with_target_only_override_accepts_cluster_object_on_scalar_dev_no() -> None:
    """A legacy ``Cluster`` target override no longer raises ``AttributeError``."""
    from qbruntime import Cluster

    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"dev_no": 3, "core_mode": "global4", "target_clusters": ["3:0"]})
    updated = spec._with(target_clusters=[Cluster.Cluster1])
    assert updated.dev_no == 3
    assert updated.clusters == ("3:1",)


def test_with_target_only_override_migrates_bare_int_cluster_using_inherited_dev_no() -> None:
    """A legacy bare-int cluster override adopts ``self.dev_no`` as the device prefix."""
    from mblt_model_zoo.utils.npu_target import NPUTargetSpec

    spec = NPUTargetSpec.from_kwargs({"dev_no": 3, "core_mode": "global4", "target_clusters": ["3:0"]})
    updated = spec._with(target_clusters=[1])
    assert updated.dev_no == 3
    assert updated.clusters == ("3:1",)


# ---------------------------------------------------------------------------
# PR #109 P2 review: public ``target_cores`` / ``target_clusters`` accessors
# must return a complete view for a multi-device backend. Previously they
# silently filtered out every device except ``_fallback_dev()``, which was
# only intended as a legacy-migration fallback.
# ---------------------------------------------------------------------------


def test_target_cores_returns_union_across_devices_on_multi_device_backend() -> None:
    """Aggregate ``target_cores`` includes entries from every device the backend covers."""
    from qbruntime import Cluster, Core, CoreId

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": [0, 1],
            "target_cores": ["0:0:0", "0:0:1", "1:0:0", "1:0:1"],
        }
    )
    assert backend.target_cores == [
        CoreId(Cluster.Cluster0, Core.Core0),
        CoreId(Cluster.Cluster0, Core.Core1),
        CoreId(Cluster.Cluster0, Core.Core0),
        CoreId(Cluster.Cluster0, Core.Core1),
    ]


def test_target_clusters_returns_union_across_devices_on_multi_device_backend() -> None:
    """Aggregate ``target_clusters`` includes entries from every device the backend covers."""
    from qbruntime import Cluster

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "global4",
            "dev_no": [0, 1],
            "target_clusters": ["0:0", "0:1", "1:0", "1:1"],
        }
    )
    assert backend.target_clusters == [
        Cluster.Cluster0,
        Cluster.Cluster1,
        Cluster.Cluster0,
        Cluster.Cluster1,
    ]


def test_target_cores_preserves_single_device_return_shape() -> None:
    """Single-device backends see the same aggregate ``target_cores`` list as before."""
    from qbruntime import Cluster, Core, CoreId

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0", "0:0:1"],
        }
    )
    assert backend.target_cores == [
        CoreId(Cluster.Cluster0, Core.Core0),
        CoreId(Cluster.Cluster0, Core.Core1),
    ]


def test_target_clusters_preserves_single_device_return_shape() -> None:
    """Single-device backends see the same aggregate ``target_clusters`` list as before."""
    from qbruntime import Cluster

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "global4",
            "dev_no": 0,
            "target_clusters": ["0:0"],
        }
    )
    assert backend.target_clusters == [Cluster.Cluster0]


def test_target_cores_by_device_groups_entries_by_device() -> None:
    """``target_cores_by_device`` maps every covered device to its CoreId list."""
    from qbruntime import Cluster, Core, CoreId

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": [0, 1],
            "target_cores": ["0:0:0", "0:0:1", "1:0:0", "1:0:1"],
        }
    )
    assert backend.target_cores_by_device == {
        0: [CoreId(Cluster.Cluster0, Core.Core0), CoreId(Cluster.Cluster0, Core.Core1)],
        1: [CoreId(Cluster.Cluster0, Core.Core0), CoreId(Cluster.Cluster0, Core.Core1)],
    }


def test_target_clusters_by_device_groups_entries_by_device() -> None:
    """``target_clusters_by_device`` maps every covered device to its Cluster list."""
    from qbruntime import Cluster

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "global4",
            "dev_no": [0, 1],
            "target_clusters": ["0:0", "0:1", "1:0", "1:1"],
        }
    )
    assert backend.target_clusters_by_device == {
        0: [Cluster.Cluster0, Cluster.Cluster1],
        1: [Cluster.Cluster0, Cluster.Cluster1],
    }


def test_target_cores_fallback_expands_clusters_on_every_covered_device() -> None:
    """When ``target_cores`` is empty, the fallback expands ``target_clusters`` across every device."""
    from qbruntime import Cluster, Core, CoreId

    # Under ``global4`` mode ``_spec.cores`` stays empty and ``_spec.clusters``
    # carries the canonical strings, so ``target_cores`` exercises the
    # cluster-expansion fallback naturally.
    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "global4",
            "dev_no": [0, 1],
            "target_clusters": ["0:0", "1:0"],
        }
    )
    assert backend._target_cores_serialized == []
    expected_cores = [Core.Core0, Core.Core1, Core.Core2, Core.Core3]
    assert backend.target_cores == [CoreId(Cluster.Cluster0, c) for c in expected_cores] * 2


def test_log_model_details_prints_per_device_breakdown_on_multi_device_backend(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``log_model_details`` prints an aggregate line and a per-device breakdown for multi-device backends."""
    from mblt_model_zoo.utils.logging import log_model_details

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": [0, 1],
            "target_cores": ["0:0:0", "1:0:0"],
        }
    )
    fake_mxq = tmp_path / "model.mxq"
    fake_mxq.write_bytes(b"\x00")
    monkeypatch.setenv("MBLT_MODEL_ZOO_VERBOSE", "1")

    log_model_details(str(fake_mxq), backend)
    out = capsys.readouterr().out
    assert "Target Cores:" in out
    assert "Device 0 Cores:" in out
    assert "Device 1 Cores:" in out


def test_log_model_details_omits_per_device_breakdown_on_single_device_backend(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Single-device backends keep the pre-fix one-line ``Target Cores:`` format."""
    from mblt_model_zoo.utils.logging import log_model_details

    backend = _load_backend(
        {
            "mxq_path": "model.mxq",
            "core_mode": "single",
            "dev_no": 0,
            "target_cores": ["0:0:0"],
        }
    )
    fake_mxq = tmp_path / "model.mxq"
    fake_mxq.write_bytes(b"\x00")
    monkeypatch.setenv("MBLT_MODEL_ZOO_VERBOSE", "1")

    log_model_details(str(fake_mxq), backend)
    out = capsys.readouterr().out
    assert "Target Cores:" in out
    assert "Device 0 Cores:" not in out
