"""Unit tests for NPU target-field normalization in ``configuration_utils``.

Covers legacy migration, ``dev_no`` sugar expansion, grain fold/unfold,
``global8`` coverage validation, and canonical round-trip via
``MobilintNPUBackend.to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import warnings

import pytest

from mblt_model_zoo.hf_transformers.utils.configuration_utils import (
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
