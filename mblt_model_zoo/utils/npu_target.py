"""Compatibility exports for NPU target topology.

New applications should import these types from :mod:`mblt_npu.npu_target`.
"""

from mblt_npu.npu_target import (
    _DEFAULT_DEV_NO,
    NPUTargetSpec,
    NPUTargetSpecPending,
    _migrate_target_clusters,
    _migrate_target_cores,
    cluster_map,
    core_map,
)

__all__ = [
    "NPUTargetSpec",
    "NPUTargetSpecPending",
    "_DEFAULT_DEV_NO",
    "_migrate_target_clusters",
    "_migrate_target_cores",
    "cluster_map",
    "core_map",
]
