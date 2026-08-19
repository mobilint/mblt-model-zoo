"""NPU target-topology spec used by :class:`MobilintNPUBackend`.

The backend previously represented its target topology as four independent
mutable fields (``dev_no``, ``core_mode``, ``target_cores``, ``target_clusters``).
HF ``from_pretrained`` applies ``model_kwargs`` via per-field ``setattr`` after
the config layer has already normalized the JSON payload, and each per-field
setter only updated the field it was named after. Reconciliation heuristics
that ran at the end of the setattr chain kept failing at new edge cases.

An intermediate refactor collapsed those four fields into a single frozen
:class:`NPUTargetSpec` and had every per-field setter atomically re-normalize
through :meth:`NPUTargetSpec.from_kwargs`. That eliminated the partial-state
race between fields but did not eliminate the *setter-order* race: each setter
ran full canonical normalization eagerly, so an intermediate spec built from
"only one field overridden so far" had to be a legal canonical form. Every
newly-discovered order interaction (dev_no-only, target-only,
core_mode-only, legacy-mixed-with-canonical, ...) landed as another
special-case branch in :meth:`_with` to compensate for the eager normalization.

This module ends that cycle by introducing :class:`NPUTargetSpecPending` — an
accumulator that captures caller intent *without* normalizing. Every per-field
override records its raw slot; nothing is validated. Normalization runs once,
in :meth:`NPUTargetSpecPending.finalize`, with the entire override picture in
hand. The four order-dependent branches of the old :meth:`_with` collapse into
a single pipeline: legacy migration → sibling drop → dev_no derive → grain
unification → off-mode drop → ``global8`` coverage validation. Setter order
becomes irrelevant because finalize sees the same accumulated state regardless
of the sequence the caller used to build it.

Override epochs: the pending accumulator is scoped to a single setter chain.
:class:`MobilintNPUBackend` promotes ``self._pending`` to a fresh baseline
(via :meth:`NPUTargetSpecPending.from_baseline`) every time it materializes
the canonical spec, so a subsequent HF setter chain — or any standalone
runtime mutation — never inherits stale intent flags from a previous chain.
Within a single chain (no accessor read between setters) accumulated
overrides finalize as one atomic decision; across chains (accessor reads
separate them) each chain sees a clean intent slate. See
:attr:`MobilintNPUBackend._spec` for the promotion callsite.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

from qbruntime import Cluster, Core, CoreId

from .core_mode import CoreMode, normalize_core_mode

# Default device index for ``dev_no`` when the caller does not pin one.
_DEFAULT_DEV_NO: int = 0

# Sentinel used to distinguish "field not overridden this session" from
# "field explicitly set to None/empty". Kept module-private and reused by
# both :class:`NPUTargetSpec` (single-call ``_with`` shim) and
# :class:`NPUTargetSpecPending` (accumulator).
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

# Authoritative validity sets for cluster / core indices. Derived from the
# maps so the two stay in sync if the Aries2 hardware topology ever changes;
# consumed by :func:`_migrate_target_cores` / :func:`_migrate_target_clusters`
# to reject caller-supplied indices at construction time before an invalid
# entry can reach :meth:`MobilintNPUBackend.filter_cores_for` (which used to
# silently drop the unknown key).
_VALID_CLUSTER_INDICES: frozenset = frozenset(cluster_map.keys())
_VALID_CORE_INDICES: frozenset = frozenset(core_map.keys())


def _check_cluster_core_indices(c_val: int, k_val: int, entry: Any) -> None:
    """Raise ``ValueError`` when ``c_val`` / ``k_val`` fall outside the Aries2 topology."""
    if c_val not in _VALID_CLUSTER_INDICES or k_val not in _VALID_CORE_INDICES:
        raise ValueError(
            f"Invalid target_cores entry {entry!r}: cluster must be in "
            f"{sorted(_VALID_CLUSTER_INDICES)} and core must be in "
            f"{sorted(_VALID_CORE_INDICES)}."
        )


def _check_cluster_index(c_val: int, entry: Any) -> None:
    """Raise ``ValueError`` when ``c_val`` falls outside the Aries2 topology."""
    if c_val not in _VALID_CLUSTER_INDICES:
        raise ValueError(
            f"Invalid target_clusters entry {entry!r}: cluster must be in {sorted(_VALID_CLUSTER_INDICES)}."
        )


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


def _dedup_dev_no(dev_no: Any) -> Any:
    """Return ``dev_no`` with duplicate device indices removed, order-preserving.

    A repeated device index in the caller's ``dev_no`` list is semantically
    equivalent to a single index — a device cannot be "used twice" — but the
    raw duplicates propagate through ``dev_no`` sugar expansion and produce
    duplicated ``target_clusters`` / ``target_cores`` entries. Downstream
    ``_validate_global8_coverage`` collapses duplicates via a set, so it
    passes on an invalid target list; the failure then surfaces as a
    confusing cluster-count assert inside
    :meth:`MobilintNPUBackend._make_slot_config`. Normalizing here keeps
    every downstream helper's contract intact.

    A list that reduces to a single unique entry is unwrapped to a scalar
    so subsequent legacy-migration heuristics treat it as a single-device
    input (the wire form ``[0, 0]`` should behave exactly like ``0``).
    Non-list inputs pass through unchanged.
    """
    if not isinstance(dev_no, (list, tuple)):
        return dev_no
    seen: set[int] = set()
    result: List[int] = []
    for d in dev_no:
        d_int = int(d)
        if d_int not in seen:
            seen.add(d_int)
            result.append(d_int)
    if len(result) == 1:
        return result[0]
    return result


def _dedup_preserve_order(entries: List[str]) -> List[str]:
    """Return ``entries`` with duplicates removed, first occurrence wins.

    A repeated canonical ``target_cores`` (``"d:c:k"``) or ``target_clusters``
    (``"d:c"``) string is semantically identical to a single occurrence — a
    target cannot be pinned twice. Raw duplicates would otherwise slip past
    :func:`_validate_global8_coverage` (which uses a set) and surface as a
    confusing cluster-count assert in
    :meth:`MobilintNPUBackend._make_slot_config`. Sibling to
    :func:`_dedup_dev_no`; keep them colocated.
    """
    seen: set[str] = set()
    result: List[str] = []
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


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
            _check_cluster_core_indices(int(parts[1]), int(parts[2]), v)
            result.append(v)
            modes.add("new")
        elif len(parts) == 2 and all(p.isdigit() for p in parts):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_cores item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c:k' form."
                )
            c_val, r_val = parts
            _check_cluster_core_indices(int(c_val), int(r_val), v)
            result.append(f"{fallback_dev}:{c_val}:{r_val}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_cores entry: {v!r}")
    if len(modes) > 1:
        raise ValueError("target_cores mixes legacy 'c:k' and canonical 'd:c:k' items; use one form for every entry.")
    return _dedup_preserve_order(result)


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
            _check_cluster_index(v, v)
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
            continue
        if not isinstance(v, str):
            raise TypeError(f"Unsupported target_clusters entry: {v!r} ({type(v).__name__})")
        parts = v.split(":")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            _check_cluster_index(int(parts[1]), v)
            result.append(v)
            modes.add("new")
        elif len(parts) == 1 and parts[0].isdigit():
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_clusters item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c' form."
                )
            _check_cluster_index(int(parts[0]), v)
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_clusters entry: {v!r}")
    if len(modes) > 1:
        raise ValueError("target_clusters mixes legacy and canonical items; use one form for every entry.")
    return _dedup_preserve_order(result)


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
    observe a partial-state moment where they disagree. Fresh specs come from
    :meth:`from_kwargs` (config-layer load, where the whole picture is present
    at once); the backend's per-field setter chain routes through
    :class:`NPUTargetSpecPending` so normalization is deferred until every
    override has been recorded.

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
    # Accumulated override history from a per-field setter chain. Excluded
    # from equality/hash/repr so two specs with the same canonical fields
    # compare equal regardless of how the caller built them. ``None`` on a
    # spec produced by :meth:`from_kwargs` (a "root" spec); populated on a
    # spec produced by :meth:`_with` so subsequent chained ``_with`` calls
    # can accumulate the raw overrides.
    _pending: Optional["NPUTargetSpecPending"] = field(default=None, compare=False, hash=False, repr=False)

    @property
    def _dev_no_overridden(self) -> bool:
        """Return ``True`` when any prior :meth:`_with` call touched ``dev_no``.

        Computed from the accumulated :class:`NPUTargetSpecPending` history.
        Root specs (no override history) return ``False``.
        """
        return self._pending is not None and self._pending.raw_dev_no is not _UNSET

    @property
    def _targets_overridden(self) -> bool:
        """Return ``True`` when any prior :meth:`_with` call touched a target grain.

        Computed from the accumulated :class:`NPUTargetSpecPending` history.
        Root specs (no override history) return ``False``.
        """
        if self._pending is None:
            return False
        return self._pending.raw_cores is not _UNSET or self._pending.raw_clusters is not _UNSET

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any], prefix: str = "") -> "NPUTargetSpec":
        """Normalize ``kwargs`` into a fully-consistent spec.

        Reads ``{prefix}dev_no`` / ``{prefix}core_mode`` /
        ``{prefix}target_cores`` / ``{prefix}target_clusters`` from
        ``kwargs`` and mutates them in place: legacy inputs are rewritten to
        canonical form, off-mode grain is dropped, and ``dev_no`` sugar is
        expanded when both target lists are absent. See ``AGENTS.md`` and
        ``CLAUDE.md`` "Transformers and MeloTTS" notes for the full
        behavioral contract.

        This is the config-layer entry point (JSON load), where every field is
        present at once and eager normalization is unambiguous. The per-field
        setter chain uses :class:`NPUTargetSpecPending` instead so that
        intermediate states between setters do not have to be independently
        legal canonical forms.

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
        dev_no = _dedup_dev_no(kwargs.get(dev_no_key, _DEFAULT_DEV_NO))
        dev_no_is_list = isinstance(dev_no, (list, tuple))
        dev_list = _normalize_dev_list(dev_no)
        fallback_dev = dev_list[0]
        dev_no_given = dev_no_key in kwargs

        raw_cores = kwargs.get(cores_key)
        raw_clusters = kwargs.get(clusters_key)

        cores, clusters = _resolve_targets(
            core_mode=core_mode,
            dev_list=dev_list,
            fallback_dev=fallback_dev,
            dev_no_is_list=dev_no_is_list,
            dev_no_given=dev_no_given,
            raw_cores=list(raw_cores) if raw_cores else [],
            raw_clusters=list(raw_clusters) if raw_clusters else [],
        )

        # Mutate the caller's ``kwargs`` dict in place for downstream code
        # that reads back the canonicalized values.
        if cores:
            kwargs[cores_key] = cores
        else:
            kwargs.pop(cores_key, None)
        if clusters:
            kwargs[clusters_key] = clusters
        else:
            kwargs.pop(clusters_key, None)

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
        """Return a new spec that applies one or more explicit overrides.

        Routes through :class:`NPUTargetSpecPending`: overrides accumulate as
        raw slots on a pending value derived from ``self``, and
        :meth:`NPUTargetSpecPending.finalize` runs the single normalization
        pipeline once every recorded override is visible. Chained calls
        (``spec._with(a=x)._with(b=y)``) extend the same pending instead of
        collapsing to two independent normalizations, so setter order
        becomes irrelevant.

        Args:
            dev_no: New ``dev_no`` value, or :data:`_UNSET` to leave the
                current pending slot alone.
            core_mode: New ``core_mode`` value, or :data:`_UNSET`.
            target_cores: New ``target_cores`` list, or :data:`_UNSET`.
            target_clusters: New ``target_clusters`` list, or :data:`_UNSET`.

        Returns:
            A new canonical :class:`NPUTargetSpec` reflecting the overrides.
            The returned spec carries the accumulated :class:`NPUTargetSpecPending`
            so a subsequent :meth:`_with` call sees the full override history.

        Raises:
            ValueError: When the accumulated overrides violate a canonical
                invariant (mixed legacy items, incomplete global8 coverage,
                caller-explicit dev_no disagreeing with caller-explicit
                target device set, ...). Raises originate from
                :meth:`NPUTargetSpecPending.finalize`.
        """
        if self._pending is None:
            base_pending = NPUTargetSpecPending(baseline=replace(self, _pending=None))
        else:
            base_pending = self._pending
        new_pending = base_pending._with(
            dev_no=dev_no,
            core_mode=core_mode,
            target_cores=target_cores,
            target_clusters=target_clusters,
        )
        return new_pending.finalize()

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


def _resolve_targets(
    *,
    core_mode: CoreMode,
    dev_list: List[int],
    fallback_dev: int,
    dev_no_is_list: bool,
    dev_no_given: bool,
    raw_cores: List[Any],
    raw_clusters: List[Any],
) -> tuple[List[str], List[str]]:
    """Canonicalize a ``(dev_no, core_mode, cores, clusters)`` payload.

    Shared by :meth:`NPUTargetSpec.from_kwargs` (config-layer load with the
    complete picture) and :meth:`NPUTargetSpecPending.finalize` (per-field
    setter chain with pre-migrated grain). Runs the ordered pipeline:

    1. Legacy migration on both raw grain lists using ``fallback_dev``.
    2. ``dev_no`` sugar expansion when both grain lists are empty.
    3. Grain unification per ``core_mode`` (unfold clusters to cores under
       ``single``; fold cores to clusters under ``multi`` / ``global4`` /
       ``global8``).
    4. Off-mode grain drop.
    5. Device-set consistency check when ``dev_no_given=True``.
    6. ``global8`` coverage validation.

    Returns:
        ``(cores, clusters)`` — canonical string lists after the pipeline.

    Raises:
        ValueError: For mixed legacy/canonical items, list-shaped ``dev_no``
            combined with legacy items, device-set mismatch, or incomplete
            ``global8`` coverage.
        TypeError: When a target entry has an unsupported type.
    """
    cores = _migrate_target_cores(raw_cores, fallback_dev, dev_no_is_list) if raw_cores else []
    clusters = _migrate_target_clusters(raw_clusters, fallback_dev, dev_no_is_list) if raw_clusters else []

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
                clusters = _fold_cores_to_clusters(cores, stacklevel=5)

        # Drop the grain that does not match core_mode. When both raw
        # fields were provided, only the mode-appropriate one is
        # authoritative; leaving the stale field in place would pollute the
        # device-set check and the backend's per-slot dispatch.
        if core_mode == "single":
            clusters = []
        else:
            cores = []

        # Device-set consistency check when the caller explicitly set
        # ``dev_no``. When the caller-explicit target set disagrees with the
        # caller-explicit ``dev_no``, the mismatch is genuine and must
        # surface as a hard error rather than a silent re-target.
        if dev_no_given:
            target_devs = _devices_from_targets(cores, clusters)
            explicit_devs = set(dev_list)
            if target_devs != explicit_devs:
                raise ValueError(
                    f"target device set {sorted(target_devs)} does not match dev_no {sorted(explicit_devs)}."
                )

    if core_mode == "global8":
        _validate_global8_coverage(clusters)

    return cores, clusters


@dataclass(frozen=True)
class NPUTargetSpecPending:
    """Accumulator for :class:`NPUTargetSpec` overrides applied one field at a time.

    HF ``from_pretrained`` fires per-field property setters (``dev_no``,
    ``core_mode``, ``target_cores``, ``target_clusters``) in an order it
    chooses, not one we control. Eagerly normalizing on every setter call
    forces intermediate spec states to be legal canonical forms — but the
    *correct* canonical form depends on future setter calls that have not
    happened yet, so early normalization is lossy (global8 coverage checks
    fire on partial clusters, device-set consistency fails against stale
    prefixes, and so on).

    :class:`NPUTargetSpecPending` sidesteps that by capturing each override
    as a raw slot without validation. :meth:`finalize` runs the single
    normalization pipeline once every accumulated override is visible.
    Setter order becomes irrelevant because the finalize step sees the same
    state regardless of the sequence the caller used to build it.

    Attributes:
        baseline: The canonical :class:`NPUTargetSpec` present before any
            setter override applied — typically the config-layer load
            result. Serves as the fallback for fields the caller never
            overrides this session.
        raw_dev_no: The caller's raw ``dev_no`` override in the original
            wire form (``int``, ``list[int]``, or :class:`~qbruntime.CoreId`-
            adjacent legacy). :data:`_UNSET` when the caller never touched
            ``dev_no``.
        raw_core_mode: The caller's raw ``core_mode`` override, or
            :data:`_UNSET`.
        raw_cores: The caller's raw ``target_cores`` override (possibly a
            list of :class:`~qbruntime.CoreId` objects, legacy 2-part
            strings, or canonical ``"d:c:k"`` strings), or :data:`_UNSET`.
        raw_clusters: The caller's raw ``target_clusters`` override, or
            :data:`_UNSET`.
    """

    baseline: NPUTargetSpec
    raw_dev_no: Any = _UNSET
    raw_core_mode: Any = _UNSET
    raw_cores: Any = _UNSET
    raw_clusters: Any = _UNSET

    @classmethod
    def from_baseline(cls, spec: NPUTargetSpec) -> "NPUTargetSpecPending":
        """Return a fresh pending baseline for the next override epoch.

        :class:`MobilintNPUBackend` calls this immediately after materializing
        its canonical spec so the next per-field setter chain accumulates on a
        clean slate. All four intent flags (:attr:`raw_dev_no`,
        :attr:`raw_core_mode`, :attr:`raw_cores`, :attr:`raw_clusters`) reset
        to :data:`_UNSET`; the caller-facing baseline for the fresh epoch is
        the canonical result of the previous epoch's finalize, with any
        accumulated ``_pending`` history stripped from the baseline value so
        the epochs are fully independent.

        Args:
            spec: Canonical :class:`NPUTargetSpec` to seed the fresh pending's
                baseline with. Typically the finalized result of the previous
                override epoch.

        Returns:
            A new :class:`NPUTargetSpecPending` whose baseline is ``spec``
            (with any prior ``_pending`` history stripped) and whose intent
            slots are all :data:`_UNSET`.
        """
        return cls(baseline=replace(spec, _pending=None))

    def _with(
        self,
        *,
        dev_no: Any = _UNSET,
        core_mode: Any = _UNSET,
        target_cores: Any = _UNSET,
        target_clusters: Any = _UNSET,
    ) -> "NPUTargetSpecPending":
        """Return a new pending with one or more raw overrides recorded.

        Pure state accumulation; no normalization, no validation. The
        finalize step consumes the entire accumulated history at once.

        Args:
            dev_no: New raw ``dev_no`` slot value, or :data:`_UNSET` to
                leave the current slot alone.
            core_mode: New raw ``core_mode`` slot value, or :data:`_UNSET`.
            target_cores: New raw ``target_cores`` slot value, or
                :data:`_UNSET`.
            target_clusters: New raw ``target_clusters`` slot value, or
                :data:`_UNSET`.

        Returns:
            A new :class:`NPUTargetSpecPending` reflecting the overrides.
        """
        return replace(
            self,
            raw_dev_no=dev_no if dev_no is not _UNSET else self.raw_dev_no,
            raw_core_mode=core_mode if core_mode is not _UNSET else self.raw_core_mode,
            raw_cores=target_cores if target_cores is not _UNSET else self.raw_cores,
            raw_clusters=target_clusters if target_clusters is not _UNSET else self.raw_clusters,
        )

    def finalize(self) -> NPUTargetSpec:
        """Compute the canonical :class:`NPUTargetSpec` from accumulated overrides.

        Runs the ordered pipeline once with every recorded override
        visible:

        1. Resolve effective ``core_mode`` (raw override or baseline).
        2. Determine the inherited ``dev_no`` (raw override, else baseline)
           and use its scalar as the legacy-migration fallback prefix.
        3. Sibling drop: if the caller overrode exactly one grain, discard
           the baseline's other grain (it is stale wrt the new authoritative
           grain). If the caller overrode neither grain but overrode
           ``dev_no`` or changed ``core_mode``, clear the baseline grain so
           :func:`_resolve_targets` re-expands from ``dev_no`` sugar.
        4. Legacy migration on the effective raw grain using the fallback.
        5. Hand the canonicalized payload to :func:`_resolve_targets` for
           the shared grain-unification / off-mode drop / consistency /
           coverage pipeline.
        6. Derive ``dev_no`` from the canonical targets when the caller did
           not pin one explicitly.

        Returns:
            A canonical :class:`NPUTargetSpec`. Its ``_pending`` field
            points at ``self`` so a subsequent chained :meth:`NPUTargetSpec._with`
            call sees the full override history.

        Raises:
            ValueError: For canonical-form invariants (mixed legacy items,
                device-set mismatch when both ``dev_no`` and targets are
                caller-explicit, incomplete global8 coverage).
            TypeError: When a raw grain entry has an unsupported type.
        """
        baseline = self.baseline
        core_mode_overridden = self.raw_core_mode is not _UNSET
        dev_no_overridden = self.raw_dev_no is not _UNSET
        cores_overridden = self.raw_cores is not _UNSET
        clusters_overridden = self.raw_clusters is not _UNSET

        core_mode: CoreMode = normalize_core_mode(self.raw_core_mode) if core_mode_overridden else baseline.core_mode

        # Inherited dev_no is what a legacy 2-part grain item should adopt
        # as its device prefix when no explicit dev_no override is in play.
        # The caller's raw ``dev_no`` wins if set; otherwise use the
        # baseline's canonical dev_no. Dedup here so a caller-supplied list
        # like ``[0, 0]`` collapses to a single-device input before it
        # reaches sugar expansion or the device-set consistency check.
        if dev_no_overridden:
            inherited_dev_no = _dedup_dev_no(self.raw_dev_no)
        else:
            inherited_dev_no = baseline.dev_no_public()
        inherited_is_list = isinstance(inherited_dev_no, (list, tuple))
        inherited_dev_list = _normalize_dev_list(inherited_dev_no)
        fallback_dev = inherited_dev_list[0]

        # Sibling-drop: pick the authoritative grain(s) with the caller's
        # accumulated intent visible.
        if cores_overridden or clusters_overridden:
            # Caller named at least one grain explicitly — that grain wins,
            # the baseline sibling is stale.
            raw_cores = list(self.raw_cores) if cores_overridden and self.raw_cores else []
            raw_clusters = list(self.raw_clusters) if clusters_overridden and self.raw_clusters else []
        elif dev_no_overridden or (core_mode_overridden and core_mode != baseline.core_mode):
            # Sugar-related override without explicit grain authority. The
            # baseline's canonical grain reflects the previous ``dev_no`` /
            # ``core_mode`` epoch and cannot be re-used verbatim (e.g. a
            # single-cluster ``target_cores`` folds to a single-cluster
            # ``target_clusters`` that fails ``global8`` coverage). Drop
            # both grain lists so :func:`_resolve_targets` re-expands from
            # the effective ``dev_no`` sugar under the new mode.
            raw_cores = []
            raw_clusters = []
        else:
            # Nothing forces the caller's hand; the baseline is still
            # canonical for the effective ``(dev_no, core_mode)`` pair.
            raw_cores = list(baseline.cores)
            raw_clusters = list(baseline.clusters)

        cores, clusters = _resolve_targets(
            core_mode=core_mode,
            dev_list=inherited_dev_list,
            fallback_dev=fallback_dev,
            dev_no_is_list=inherited_is_list,
            dev_no_given=dev_no_overridden,
            raw_cores=raw_cores,
            raw_clusters=raw_clusters,
        )

        # Resolve the effective ``dev_no`` for the returned canonical spec.
        # Caller override wins (reusing the already-deduped ``inherited_dev_no``
        # so a caller-supplied list like ``[0, 0]`` collapses to a scalar);
        # otherwise derive from the canonical target device prefixes so the
        # spec is self-consistent for a later setter-chain override.
        if dev_no_overridden:
            resolved_dev_no: Any = inherited_dev_no
        elif cores or clusters:
            derived_devs = sorted(_devices_from_targets(cores, clusters))
            resolved_dev_no = derived_devs if len(derived_devs) > 1 else derived_devs[0]
        else:
            resolved_dev_no = baseline.dev_no_public()

        return NPUTargetSpec(
            dev_no=_freeze_dev_no(resolved_dev_no),
            core_mode=core_mode,
            cores=tuple(cores),
            clusters=tuple(clusters),
            _pending=self,
        )


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
