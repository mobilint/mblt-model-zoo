"""Frozen NPU target-topology spec used by :class:`MobilintNPUBackend`.

The backend previously represented its target topology as four independent
mutable fields (``dev_no``, ``core_mode``, ``target_cores``, ``target_clusters``).
HF ``from_pretrained`` applies ``model_kwargs`` via per-field ``setattr`` after
the config layer has already normalized the JSON payload, and each per-field
setter only updated the field it was named after. Reconciliation heuristics
that ran at the end of the setattr chain kept failing at new edge cases.

This module collapses those four fields into a single frozen dataclass —
:class:`NPUTargetSpec` — so no partial-state moment can exist. Every setter
on the backend re-derives the whole spec through :meth:`NPUTargetSpec._with`,
which forwards to the canonical :meth:`NPUTargetSpec.from_kwargs` normalizer.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Union

from qbruntime import Cluster, Core, CoreId

from .core_mode import CoreMode, normalize_core_mode

# Default device index for ``dev_no`` when the caller does not pin one.
_DEFAULT_DEV_NO: int = 0

# Sentinel used by :meth:`NPUTargetSpec._with` to distinguish "field not
# provided" from "field explicitly set to None/empty". Kept module-private.
_UNSET: Any = object()


cluster_map: Dict[int, "Cluster"] = {
    0: Cluster.Cluster0,
    1: Cluster.Cluster1,
}

core_map: Dict[int, "Core"] = {
    0: Core.Core0,
    1: Core.Core1,
    2: Core.Core2,
    3: Core.Core3,
}

# Inverse maps: ``qbruntime`` enums do not expose a numeric conversion (their
# ``.value`` returns the enum itself), so we build inverse lookups here and
# reuse them everywhere a ``Cluster`` / ``Core`` object must be serialized to
# its integer index.
_cluster_int_map: Dict["Cluster", int] = {v: k for k, v in cluster_map.items()}
_core_int_map: Dict["Core", int] = {v: k for k, v in core_map.items()}


def cluster_to_int(cluster: "Cluster") -> int:
    """Return the integer index for a ``qbruntime.Cluster`` enum member."""
    return _cluster_int_map[cluster]


def core_to_int(core: "Core") -> int:
    """Return the integer index for a ``qbruntime.Core`` enum member."""
    return _core_int_map[core]


def _normalize_dev_list(dev_no: Any) -> List[int]:
    """Coerce a ``dev_no`` sugar value into a list of device indices."""
    if isinstance(dev_no, (list, tuple)):
        return [int(d) for d in dev_no]
    return [int(dev_no)]


def _migrate_target_cores(values: list, fallback_dev: int, dev_no_is_list: bool) -> List[str]:
    """Migrate a mixed ``target_cores`` list to the canonical ``"d:c:k"`` form.

    Args:
        values: Raw ``target_cores`` list. Items may be ``CoreId`` objects,
            fully-qualified ``"d:c:k"`` strings, or legacy ``"c:k"`` strings.
        fallback_dev: Device prefix to apply to legacy items when ``dev_no``
            is a scalar.
        dev_no_is_list: ``True`` if the caller passed a list-shaped ``dev_no``.
            Legacy items (both string-form ``"c:k"`` and ``CoreId`` objects)
            lack a device prefix and are ambiguous in that case; both are
            rejected symmetrically.

    Returns:
        A list of canonical ``"d:c:k"`` strings.

    Raises:
        ValueError: If entries mix legacy and new forms, or if a legacy entry
            (string or :class:`~qbruntime.CoreId`) appears with a list-shaped
            ``dev_no``.
        TypeError: If an entry is neither ``CoreId`` nor a string.
    """
    result: List[str] = []
    modes: set[str] = set()
    for v in values:
        if isinstance(v, CoreId):
            if dev_no_is_list:
                raise ValueError(
                    "Legacy CoreId target_cores entry "
                    f"{cluster_to_int(v.cluster)}:{core_to_int(v.core)} is ambiguous "
                    "when dev_no is a list; use the fully-qualified 'd:c:k' string form."
                )
            result.append(f"{fallback_dev}:{cluster_to_int(v.cluster)}:{core_to_int(v.core)}")
            modes.add("legacy")
            continue
        if not isinstance(v, str):
            raise TypeError(f"Unsupported target_cores entry: {v!r} ({type(v).__name__})")
        parts = v.split(":")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            result.append(v)
            modes.add("new")
        elif len(parts) == 2 and all(p.isdigit() for p in parts):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_cores item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c:k' form."
                )
            c_val, r_val = parts
            result.append(f"{fallback_dev}:{c_val}:{r_val}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_cores entry: {v!r}")
    if len(modes) > 1:
        raise ValueError("target_cores mixes legacy 'c:k' and canonical 'd:c:k' items; use one form for every entry.")
    return result


def _migrate_target_clusters(values: list, fallback_dev: int, dev_no_is_list: bool) -> List[str]:
    """Migrate a mixed ``target_clusters`` list to the canonical ``"d:c"`` form.

    Args:
        values: Raw ``target_clusters`` list. Items may be ``Cluster`` objects,
            fully-qualified ``"d:c"`` strings, bare integers, or bare
            ``"c"`` strings (legacy).
        fallback_dev: Device prefix to apply to legacy items when ``dev_no``
            is a scalar.
        dev_no_is_list: ``True`` if the caller passed a list-shaped ``dev_no``.
            Legacy items (``Cluster`` objects, bare ints, and bare ``"c"``
            strings) lack a device prefix and are ambiguous in that case;
            they are all rejected symmetrically.

    Returns:
        A list of canonical ``"d:c"`` strings.

    Raises:
        ValueError: If entries mix legacy and new forms, or if a legacy entry
            (``Cluster``, ``int``, or bare ``"c"`` string) appears with a
            list-shaped ``dev_no``.
        TypeError: If an entry is neither ``Cluster``, ``int``, nor a string.
    """
    result: List[str] = []
    modes: set[str] = set()
    for v in values:
        if isinstance(v, Cluster):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy Cluster target_clusters entry {cluster_to_int(v)} is "
                    "ambiguous when dev_no is a list; use the fully-qualified 'd:c' "
                    "string form."
                )
            result.append(f"{fallback_dev}:{cluster_to_int(v)}")
            modes.add("legacy")
            continue
        if isinstance(v, bool):
            raise TypeError(f"Unsupported target_clusters entry: {v!r}")
        if isinstance(v, int):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_clusters int {v} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c' form."
                )
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
            continue
        if not isinstance(v, str):
            raise TypeError(f"Unsupported target_clusters entry: {v!r} ({type(v).__name__})")
        parts = v.split(":")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            result.append(v)
            modes.add("new")
        elif len(parts) == 1 and parts[0].isdigit():
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_clusters item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c' form."
                )
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_clusters entry: {v!r}")
    if len(modes) > 1:
        raise ValueError("target_clusters mixes legacy and canonical items; use one form for every entry.")
    return result


def _expand_clusters_to_cores(clusters: List[str]) -> List[str]:
    """Expand each canonical ``"d:c"`` cluster string into its four ``"d:c:k"`` cores."""
    result: List[str] = []
    for cs in clusters:
        d_val, c_val = cs.split(":")
        for k in range(4):
            result.append(f"{d_val}:{c_val}:{k}")
    return result


def _fold_cores_to_clusters(cores: List[str], *, stacklevel: int) -> List[str]:
    """Fold canonical ``"d:c:k"`` core strings up to their unique ``"d:c"`` cluster prefixes.

    Emits a ``UserWarning`` for each cluster whose covered cores are a strict
    subset of ``{0, 1, 2, 3}`` so callers know the effective target was
    rounded up.
    """
    per_cluster: dict[tuple[int, int], set[int]] = defaultdict(set)
    order: list[tuple[int, int]] = []
    for cs in cores:
        d_val, c_val, k_val = (int(x) for x in cs.split(":"))
        key = (d_val, c_val)
        if key not in per_cluster:
            order.append(key)
        per_cluster[key].add(k_val)
    for key in order:
        ks = per_cluster[key]
        if ks != {0, 1, 2, 3}:
            warnings.warn(
                f"target_cores {sorted(ks)} for cluster {key[0]}:{key[1]} do not cover all "
                f"four cores; rounded up to whole cluster.",
                UserWarning,
                stacklevel=stacklevel,
            )
    return [f"{d_val}:{c_val}" for (d_val, c_val) in order]


def _devices_from_targets(cores: List[str], clusters: List[str]) -> set[int]:
    """Return the set of device indices referenced by any canonical target string."""
    devs: set[int] = set()
    for cs in cores:
        devs.add(int(cs.split(":", 1)[0]))
    for cs in clusters:
        devs.add(int(cs.split(":", 1)[0]))
    return devs


def _validate_global8_coverage(clusters: List[str]) -> None:
    """Ensure every unique device in ``clusters`` covers both clusters 0 and 1."""
    per_dev: dict[int, set[int]] = defaultdict(set)
    for cs in clusters:
        d_val, c_val = (int(x) for x in cs.split(":"))
        per_dev[d_val].add(c_val)
    for d_val, cs_set in per_dev.items():
        if cs_set != {0, 1}:
            raise ValueError(
                f"core_mode='global8' requires both clusters for device {d_val}; got clusters {sorted(cs_set)}."
            )


@dataclass(frozen=True)
class NPUTargetSpec:
    """Atomic representation of the NPU target topology.

    The four fields (``dev_no``, ``core_mode``, ``cores``, ``clusters``) are
    stored as one immutable value so :class:`MobilintNPUBackend` cannot
    observe a partial-state moment where they disagree. Every per-field
    setter on the backend replaces the spec whole via :meth:`_with`, which
    forwards to :meth:`from_kwargs` for full canonical renormalization.

    Attributes:
        dev_no: Canonical device index (``int``) or list of device indices
            (``tuple[int, ...]``). Derived from the canonical target strings
            when the caller does not pin it explicitly.
        core_mode: One of ``"single"``, ``"multi"``, ``"global4"``,
            ``"global8"``.
        cores: Canonical ``"d:c:k"`` strings; populated in ``"single"``
            mode.
        clusters: Canonical ``"d:c"`` strings; populated in
            ``"multi"`` / ``"global4"`` / ``"global8"`` modes.
    """

    dev_no: Union[int, tuple]
    core_mode: CoreMode
    cores: tuple = field(default_factory=tuple)
    clusters: tuple = field(default_factory=tuple)
    # Caller-intent tracking: ``True`` after any :meth:`_with` call that
    # explicitly overrode the corresponding field. Flags accumulate across
    # the HF ``setattr`` chain so a follow-up sibling override can decide
    # between "sync sibling to me" (implicit intent) and "enforce
    # consistency check" (both fields caller-explicit).
    _dev_no_overridden: bool = False
    _targets_overridden: bool = False

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any], prefix: str = "") -> "NPUTargetSpec":
        """Normalize ``kwargs`` into a fully-consistent spec.

        Reads ``{prefix}dev_no`` / ``{prefix}core_mode`` /
        ``{prefix}target_cores`` / ``{prefix}target_clusters`` from
        ``kwargs`` and mutates them in place: legacy inputs are rewritten to
        canonical form, off-mode grain is dropped, and ``dev_no`` sugar is
        expanded when both target lists are absent. See
        ``AGENTS.md`` and ``CLAUDE.md`` "Transformers and MeloTTS" notes for
        the full behavioral contract.

        Args:
            kwargs: Keyword-argument dict handed to a config mixin's
                ``__init__`` / ``__post_init__``. Target keys are mutated in
                place so callers can pass the same dict to
                :meth:`MobilintNPUBackend.from_dict`.
            prefix: Optional prefix that scopes the NPU keys
                (e.g. ``"encoder_"``, ``"vision_"``, ``"base_"``). Empty for
                the default backend.

        Returns:
            A fully-canonical :class:`NPUTargetSpec` value.

        Raises:
            ValueError: For any inconsistency the spec calls out — mixed
                legacy and new items, list-shaped ``dev_no`` combined with
                legacy items, device-set mismatch, or incomplete global8
                coverage.
            TypeError: When a target entry has an unsupported type.
        """
        core_mode_key = f"{prefix}core_mode"
        dev_no_key = f"{prefix}dev_no"
        cores_key = f"{prefix}target_cores"
        clusters_key = f"{prefix}target_clusters"

        core_mode = normalize_core_mode(kwargs.get(core_mode_key, "single"))
        dev_no = kwargs.get(dev_no_key, _DEFAULT_DEV_NO)
        dev_no_is_list = isinstance(dev_no, (list, tuple))
        dev_list = _normalize_dev_list(dev_no)
        fallback_dev = dev_list[0]
        dev_no_given = dev_no_key in kwargs

        raw_cores = kwargs.get(cores_key)
        raw_clusters = kwargs.get(clusters_key)

        cores = _migrate_target_cores(list(raw_cores), fallback_dev, dev_no_is_list) if raw_cores else []
        clusters = _migrate_target_clusters(list(raw_clusters), fallback_dev, dev_no_is_list) if raw_clusters else []

        if not cores and not clusters:
            # ``dev_no`` sugar expansion when both target lists are absent.
            if core_mode == "single":
                cores = [f"{d}:{c}:{k}" for d in dev_list for c in (0, 1) for k in range(4)]
            else:
                clusters = [f"{d}:{c}" for d in dev_list for c in (0, 1)]
        else:
            # Grain unification per core_mode.
            if core_mode == "single":
                if not cores and clusters:
                    cores = _expand_clusters_to_cores(clusters)
            else:
                if not clusters and cores:
                    clusters = _fold_cores_to_clusters(cores, stacklevel=4)

            # Drop the grain that does not match core_mode. When both raw
            # fields were provided, only the mode-appropriate one is
            # authoritative (matching the legacy ``from_dict`` warning);
            # leaving the stale field in place would pollute the device-set
            # check and the backend's per-slot dispatch.
            if core_mode == "single":
                clusters = []
                kwargs.pop(clusters_key, None)
            else:
                cores = []
                kwargs.pop(cores_key, None)

            # Device-set consistency check when the caller explicitly set
            # dev_no. When the caller-explicit target set disagrees with the
            # caller-explicit dev_no, the mismatch is genuine and must
            # surface as a hard error rather than a silent re-target.
            if dev_no_given:
                target_devs = _devices_from_targets(cores, clusters)
                explicit_devs = set(dev_list)
                if target_devs != explicit_devs:
                    raise ValueError(
                        f"target device set {sorted(target_devs)} does not match {dev_no_key} {sorted(explicit_devs)}."
                    )

        if core_mode == "global8":
            _validate_global8_coverage(clusters)

        if cores:
            kwargs[cores_key] = cores
        if clusters:
            kwargs[clusters_key] = clusters

        # When the caller supplied canonical targets but no explicit ``dev_no``,
        # derive ``dev_no`` from the target device prefixes so the in-memory
        # spec is self-consistent. Without this, a later :meth:`_with` call
        # (e.g. an isolated ``core_mode`` override) would round-trip the stale
        # default ``dev_no=0`` back through :meth:`from_kwargs`, flip
        # ``dev_no_given=True``, and fail the device-set consistency check on
        # an otherwise valid config.
        if not dev_no_given and (cores or clusters):
            derived_devs = sorted(_devices_from_targets(cores, clusters))
            dev_no = derived_devs if len(derived_devs) > 1 else derived_devs[0]

        return cls(
            dev_no=_freeze_dev_no(dev_no),
            core_mode=core_mode,
            cores=tuple(cores),
            clusters=tuple(clusters),
        )

    def _with(
        self,
        *,
        dev_no: Any = _UNSET,
        core_mode: Any = _UNSET,
        target_cores: Any = _UNSET,
        target_clusters: Any = _UNSET,
    ) -> "NPUTargetSpec":
        """Return a new spec with one or more explicit overrides.

        Atomic replace: every partial mutation triggers a full
        renormalization through :meth:`from_kwargs`, so callers never
        observe a moment where the four target fields disagree.

        Intent resolution when only one of ``{dev_no, targets}`` is
        overridden in this call and its sibling was never previously
        overridden (i.e., still reflects the initial :meth:`from_kwargs`
        load):

        - **Target-only override**: sync ``dev_no`` to the target device
          set. This is the ``--vision-target-cores 1:0:0`` path where the
          JSON's stale ``dev_no=0`` should not override the caller's
          explicit target.
        - **``dev_no``-only override**: clear the stale target lists so
          :meth:`from_kwargs` re-expands them from the new ``dev_no``
          sugar. This is the ``--dev-no 1`` path where the JSON's stale
          dev0 cores must not pin the new backend to the old device.

        When both fields have been overridden at any point in the
        ``setattr`` chain, both are treated as caller-authoritative and the
        device-set consistency check inside :meth:`from_kwargs` catches
        genuine mismatches.

        Args:
            dev_no: New ``dev_no`` value, or :data:`_UNSET` to inherit.
            core_mode: New ``core_mode`` value, or :data:`_UNSET`.
            target_cores: New ``target_cores`` list, or :data:`_UNSET`.
            target_clusters: New ``target_clusters`` list, or :data:`_UNSET`.

        Returns:
            A new canonical :class:`NPUTargetSpec` reflecting the overrides.

        Raises:
            ValueError: When ``dev_no`` and target device sets have both
                been caller-overridden and disagree, or when other
                canonical-form invariants fail (mixed legacy items,
                incomplete global8 coverage, etc.).
        """
        dev_no_changed = dev_no is not _UNSET
        core_mode_changed = core_mode is not _UNSET
        cores_changed = target_cores is not _UNSET
        clusters_changed = target_clusters is not _UNSET
        targets_changed = cores_changed or clusters_changed

        new_dev_no: Any = dev_no if dev_no_changed else self.dev_no
        new_core_mode = core_mode if core_mode_changed else self.core_mode
        new_cores: list = list(target_cores) if cores_changed else list(self.cores)
        new_clusters: list = list(target_clusters) if clusters_changed else list(self.clusters)

        # When exactly one target grain is explicitly overridden, discard the
        # sibling grain carried over from ``self`` before renormalization. The
        # sibling reflects the previous ``core_mode`` epoch and is only stale
        # intent once the caller re-authoritatively names one grain; keeping
        # it would (a) pollute the target-only ``dev_no`` sync below by
        # unioning stale device prefixes with the new grain, and (b) surface
        # as a spurious device-set mismatch inside :meth:`from_kwargs` when
        # its off-mode-grain drop then reduces the target device set. This
        # makes the setter order symmetric: applying ``target_clusters``
        # before ``core_mode`` converges on the same spec as the reverse.
        if cores_changed and not clusters_changed:
            new_clusters = []
        elif clusters_changed and not cores_changed:
            new_cores = []

        if targets_changed and not dev_no_changed and not self._dev_no_overridden:
            # Target-only override with un-overridden ``dev_no``: the
            # canonical target strings unambiguously carry the device prefix,
            # so sync ``dev_no`` to match rather than clobbering the caller's
            # explicit target.
            target_devs = _devices_from_targets(new_cores, new_clusters)
            if target_devs:
                sorted_devs = sorted(target_devs)
                new_dev_no = sorted_devs if len(sorted_devs) > 1 else sorted_devs[0]
        elif dev_no_changed and not targets_changed and not self._targets_overridden:
            # ``dev_no``-only override with un-overridden targets: clear the
            # stale target lists so :meth:`from_kwargs` re-expands them from
            # the new ``dev_no`` sugar.
            new_cores = []
            new_clusters = []
        # else: both explicitly overridden at some point (or both untouched
        # this call) — pass both through and let :meth:`from_kwargs`'s
        # consistency check catch mismatches.

        kwargs: Dict[str, Any] = {
            "dev_no": _thaw_dev_no(new_dev_no),
            "core_mode": new_core_mode,
        }
        if new_cores:
            kwargs["target_cores"] = new_cores
        if new_clusters:
            kwargs["target_clusters"] = new_clusters

        rebuilt = NPUTargetSpec.from_kwargs(kwargs)

        # Preserve caller-intent flags so subsequent :meth:`_with` calls
        # see the accumulated override history.
        return replace(
            rebuilt,
            _dev_no_overridden=self._dev_no_overridden or dev_no_changed,
            _targets_overridden=self._targets_overridden or targets_changed,
        )

    def dev_no_public(self) -> Union[int, List[int]]:
        """Return ``dev_no`` in its user-facing shape (``int`` or ``list``)."""
        return _thaw_dev_no(self.dev_no)

    def unique_devices(self) -> List[int]:
        """Return the sorted list of unique device indices referenced by targets.

        Falls back to :attr:`dev_no` sugar when both target lists are empty
        (defensive; :meth:`from_kwargs` normally guarantees at least one
        populated field).
        """
        devs = _devices_from_targets(list(self.cores), list(self.clusters))
        if devs:
            return sorted(devs)
        dev_list = _normalize_dev_list(_thaw_dev_no(self.dev_no))
        return sorted(set(dev_list)) or [0]

    def dev_no_for_serialization(self) -> Union[int, List[int]]:
        """Return ``dev_no`` derived from canonical targets for round-trip.

        When canonical target strings are set, ``dev_no`` is derived from
        their device prefixes so the emitted dict round-trips through
        :meth:`from_kwargs` — the device-set consistency check requires
        ``dev_no`` and the target device set to agree once both are
        explicit. A single device collapses to an ``int``; multiple
        devices emit a sorted ``list``. When no targets are set (early
        construction before sugar expansion), the stored :attr:`dev_no`
        is passed through as-is.
        """
        if self.cores or self.clusters:
            devs = _devices_from_targets(list(self.cores), list(self.clusters))
            if devs:
                sorted_devs = sorted(devs)
                return sorted_devs if len(sorted_devs) > 1 else sorted_devs[0]
        return _thaw_dev_no(self.dev_no)


def _freeze_dev_no(dev_no: Any) -> Union[int, tuple]:
    """Freeze a ``dev_no`` value so it can live inside a frozen dataclass."""
    if isinstance(dev_no, (list, tuple)):
        return tuple(int(d) for d in dev_no)
    return int(dev_no)


def _thaw_dev_no(dev_no: Any) -> Union[int, List[int]]:
    """Convert a frozen ``dev_no`` value back into its list-shaped counterpart."""
    if isinstance(dev_no, tuple):
        return list(dev_no)
    if isinstance(dev_no, list):
        return list(dev_no)
    return int(dev_no)
