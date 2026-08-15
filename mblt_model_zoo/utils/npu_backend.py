"""NPU backend implementation for Mobilint hardware accelerators.

Provides the :class:`MobilintNPUBackend` class which wraps the ``qbruntime``
library to load, configure, and run MXQ models on Mobilint NPU devices.

The backend can host ``N`` :class:`~qbruntime.Model` instances across one or
more :class:`~qbruntime.Accelerator` handles. ``N`` is derived from
``max_batch_size`` and the compiled batch axis ``K`` probed from the first
loaded slot (``N = ceil(max_batch_size / K)``). Slots are distributed across
the unique devices referenced by the canonical target strings in round-robin
order, and per-device accelerators are shared. ``Model.infer`` is blocking, so
callers that want concurrent NPU utilization thread their dispatches across
:meth:`MobilintNPUBackend.infer_slot` calls.

Target-topology fields (``dev_no`` / ``core_mode`` / ``target_cores`` /
``target_clusters``) live inside a single immutable :class:`NPUTargetSpec` on
``self._spec``. Every per-field setter atomically replaces ``_spec`` via
:meth:`NPUTargetSpec._with`, so no partial-state moment can exist between HF
``setattr`` calls in ``from_pretrained``.

Backwards compatibility: for callers written against a single ``Model`` /
``Accelerator`` handle, :attr:`~MobilintNPUBackend.mxq_model` and
:attr:`~MobilintNPUBackend.acc` remain accessible and refer to the first slot.
"""

import logging
import math
import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig, QbRuntimeError

from .core_mode import CoreMode, normalize_core_mode
from .logging import log_model_details
from .npu_target import (
    NPUTargetSpec,
    cluster_map,
    core_map,
)

logger = logging.getLogger(__name__)


def _is_qbruntime_bad_alloc(exc: BaseException) -> bool:
    """Return True when ``exc`` looks like a device-memory ``BadAlloc`` failure.

    ``qbruntime`` exposes a single :class:`~qbruntime.QbRuntimeError` class for
    every runtime failure it can raise (device-memory ``BadAlloc``, invalid
    MXQ artifact, incompatible target configuration, corrupted artifact,
    missing runtime dependency, ...). The ``BadAlloc`` signal only lives
    inside the error message, so we detect it by looking for the
    ``BadAlloc`` token case-insensitively and ignoring interior whitespace to
    be resilient to slight formatting differences across ``qbruntime``
    versions. Isolated as a helper so a future ``BadAllocError`` subclass can
    replace this check in one place.
    """
    message = str(exc) if exc is not None else ""
    return "badalloc" in message.lower().replace(" ", "")


class MobilintBackendAllocError(RuntimeError):
    """Raised when a multi-slot backend fails to create or launch a slot.

    Only fires for ``qbruntime`` device-memory ``BadAlloc`` failures. Any
    other :class:`~qbruntime.QbRuntimeError` (invalid MXQ, bad target
    configuration, corrupted artifact, missing runtime dependency, ...) is
    re-raised unchanged from :meth:`MobilintNPUBackend.create` and
    :meth:`MobilintNPUBackend.launch` so callers can distinguish a memory
    ceiling from a user-config or artifact bug.

    Carries enough context (phase, slot index, device, how many slots had
    already succeeded, current sizing knobs) to help the caller locate the
    device memory boundary and pick a safer ``max_batch_size``.

    Attributes:
        phase: ``"create"`` when :func:`~qbruntime.Model` construction failed,
            ``"launch"`` when :meth:`~qbruntime.Model.launch` failed.
        slot: Zero-indexed slot at which the failure happened.
        dev: Device number that the failing slot was assigned to.
        succeeded_so_far: Number of slots that completed the same phase
            before the failure.
        n_total: Planned total slot count for this backend.
        max_batch_size: The ``max_batch_size`` requested by the caller.
        k_per_model: The compiled batch axis ``K`` probed from slot 0. May
            be ``1`` when the slot 0 probe itself failed.
        original: The original :class:`~qbruntime.QbRuntimeError` that fired.
    """

    def __init__(
        self,
        phase: str,
        slot: int,
        dev: int,
        succeeded_so_far: int,
        n_total: int,
        max_batch_size: int,
        k_per_model: int,
        original: BaseException,
    ) -> None:
        self.phase = phase
        self.slot = slot
        self.dev = dev
        self.succeeded_so_far = succeeded_so_far
        self.n_total = n_total
        self.max_batch_size = max_batch_size
        self.k_per_model = k_per_model
        self.original = original
        message = (
            f"[Mobilint] NPU backend {phase} failed at slot {slot} on device {dev} "
            f"(succeeded {succeeded_so_far}/{n_total}). "
            f"max_batch_size={max_batch_size}, k_per_model={k_per_model}. "
            f"Original qbruntime error: {original}. "
            "If this is a BadAlloc, lower max_batch_size or spread the workload across more devices."
        )
        super().__init__(message)


class MobilintNPUBackend:
    """Backend that runs one or more MXQ models on Mobilint NPU devices.

    Wraps the ``qbruntime`` ``Model`` and ``Accelerator`` APIs and provides
    helpers for locating MXQ model files either locally or on HuggingFace Hub.
    A single backend instance manages up to ``N`` model slots across one or
    more accelerators; slot 0 is the compatibility default consumed by
    :meth:`__call__`, :meth:`get_dtype`, and :meth:`get_input_buffer_info`.

    Class Attributes:
        num_of_clusters: Total number of hardware clusters available per device.
        num_of_cores_in_cluster: Number of cores per cluster.
    """

    num_of_clusters = 2
    num_of_cores_in_cluster = 4

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: Optional[Union[int, List[int]]] = None,
        max_batch_size: int = 1,
        core_mode: CoreMode = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[List[Union[int, str, "Cluster"]]] = None,
        revision: Optional[str] = None,
        commit_hash: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the NPU backend configuration.

        Args:
            mxq_path: Path to the compiled MXQ model file.
            dev_no: Accelerator device number(s). Accepts either a single
                index or a list of indices. Callers that pass the fully
                qualified target strings (``"d:c:k"`` / ``"d:c"``) may
                also pass a list here to declare the covered device set.
                Otherwise ``dev_no`` acts as syntactic sugar: it is
                expanded into ``target_cores`` / ``target_clusters`` when
                those lists are empty, and prepends the device prefix to
                legacy 2-part items.
            max_batch_size: Requested aggregate batch capacity. The backend
                launches enough slots so that ``N * K >= max_batch_size``,
                where ``K`` is the compiled batch axis of the MXQ artifact.
            core_mode: Execution mode that determines how NPU cores are
                allocated. One of ``"single"``, ``"multi"``, ``"global4"``,
                or ``"global8"``.
            target_cores: List of core identifiers used in ``"single"``
                mode. The canonical form is a fully-qualified
                ``"d:c:k"`` string (device : cluster : core). Legacy
                ``"c:k"`` strings and :class:`~qbruntime.CoreId` objects
                are accepted and rewritten to canonical form using
                ``dev_no`` as the device prefix. ``None`` leaves the
                configuration to be filled by ``dev_no`` sugar.
            target_clusters: List of cluster identifiers used in
                ``"multi"``, ``"global4"``, and ``"global8"`` modes. The
                canonical form is a fully-qualified ``"d:c"`` string.
                Legacy integers, :class:`~qbruntime.Cluster` objects, and
                bare ``"c"`` strings are accepted and rewritten to
                canonical form using ``dev_no`` as the device prefix.
            revision: HuggingFace Hub revision (branch, tag, or commit SHA)
                to use when downloading the model file.
            commit_hash: Explicit commit hash for the Hub revision.
            **kwargs: Additional keyword arguments (ignored; kept for
                forward-compatibility).
        """
        self.name_or_path: str = ""  # will be populated in MobilintModelMixin
        self.revision = revision
        self._commit_hash = commit_hash
        self.mxq_path = mxq_path
        self.max_batch_size = max(1, max_batch_size)

        # Multi-slot backing state; populated in create()/launch().
        # ``self.acc`` and ``self.mxq_model`` remain accessible as
        # compatibility properties that read the first slot.
        self.accs: Dict[int, "Accelerator"] = {}
        self.mxq_models: List["Model"] = []
        self.model_dev_no: List[int] = []
        self.k_per_model: int = 1
        self.n_models: int = 0
        # Cached batched-infer output layout. Populated lazily by
        # :attr:`output_layout` from the compiled MXQ shape probe or the
        # runtime fallback in :mod:`multi_slot_dispatch`.
        self._output_layout_cached: Optional[Literal["n_items", "n_tokens"]] = None
        # Cached :class:`MultiSlotDispatcher` bound to this backend. Populated
        # lazily by :attr:`dispatcher` so callers can import the backend
        # without pulling the ``hf_transformers`` package into the compile-only
        # path.
        self._dispatcher: Optional[Any] = None

        # Collapse the four target-topology fields into a single frozen
        # :class:`NPUTargetSpec`. Every per-field setter below rebuilds
        # ``_spec`` via :meth:`NPUTargetSpec._with`, which forwards to
        # :meth:`NPUTargetSpec.from_kwargs` for full canonical
        # renormalization. This eliminates the partial-state race that
        # per-field setattr chains used to expose.
        #
        # ``dev_no=None`` means "not given by the caller" — the sentinel
        # keeps :meth:`NPUTargetSpec.from_kwargs` from running its
        # device-set consistency check against a defaulted ``dev_no`` when
        # the caller only supplied ``target_cores`` / ``target_clusters``.
        spec_kwargs: Dict[str, Any] = {"core_mode": normalize_core_mode(core_mode)}
        if dev_no is not None:
            spec_kwargs["dev_no"] = dev_no
        if target_cores is not None:
            spec_kwargs["target_cores"] = list(target_cores)
        if target_clusters is not None:
            spec_kwargs["target_clusters"] = list(target_clusters)
        self._spec: NPUTargetSpec = NPUTargetSpec.from_kwargs(spec_kwargs)

    # ---- Target-topology accessors ------------------------------------------
    #
    # Every setter atomically replaces ``self._spec`` through
    # :meth:`NPUTargetSpec._with`. HF ``from_pretrained`` fires these setters
    # one field at a time via ``model_kwargs`` application; the atomic replace
    # guarantees every intermediate state remains fully canonical, so no
    # snapshot-diff reconciliation pass is required.

    @property
    def dev_no(self) -> Union[int, List[int]]:
        """User-facing ``dev_no`` (``int`` or ``list[int]``)."""
        return self._spec.dev_no_public()

    @dev_no.setter
    def dev_no(self, value: Union[int, List[int]]) -> None:
        self._spec = self._spec._with(dev_no=value)

    @property
    def core_mode(self) -> CoreMode:
        return self._spec.core_mode

    @core_mode.setter
    def core_mode(self, value: str) -> None:
        self._spec = self._spec._with(core_mode=normalize_core_mode(value))

    @property
    def _target_cores_serialized(self) -> List[str]:
        """Canonical ``"d:c:k"`` strings (read-only view backed by ``_spec``)."""
        return list(self._spec.cores)

    @property
    def _target_clusters_serialized(self) -> List[str]:
        """Canonical ``"d:c"`` strings (read-only view backed by ``_spec``)."""
        return list(self._spec.clusters)

    def check_model_path(self, mxq_path: str) -> str:
        """Resolves the absolute path to an MXQ model file.

        Resolution is attempted in the following order:

        1. The path exists as-is (relative or absolute).
        2. The path exists relative to ``self.name_or_path`` (local directory).
        3. The file is downloaded from HuggingFace Hub.

        Args:
            mxq_path: Filename or relative path of the MXQ model to locate.

        Returns:
            The resolved absolute path to the MXQ file.

        Raises:
            EntryNotFoundError: If the file cannot be found on HuggingFace Hub
                after all fallback strategies are exhausted.
            Exception: If no strategy succeeds in locating the file.
        """
        # 1. current relative/absolute path
        if os.path.exists(mxq_path):
            return mxq_path

        # 2. inside the local path
        if os.path.isdir(self.name_or_path):
            local_path = os.path.join(self.name_or_path, mxq_path)
            if os.path.exists(local_path):
                return local_path

        # 3. If none of above, download mxq file from hub
        else:
            name_or_path = (
                self.name_or_path if self.name_or_path.startswith("mobilint/") else "mobilint/" + self.name_or_path
            )
            revision = (
                getattr(self, "revision", None)
                or getattr(self, "_commit_hash", None)
                or self._infer_hf_revision_from_cache(name_or_path)
            )
            try:
                return hf_hub_download(
                    repo_id=name_or_path,
                    filename=mxq_path,
                    revision=revision,
                )
            except EntryNotFoundError:
                try:
                    return hf_hub_download(
                        repo_id=name_or_path,
                        filename=mxq_path,
                    )
                except EntryNotFoundError:
                    mxq_revision = self._infer_revision_from_mxq_path(mxq_path)
                    if mxq_revision and mxq_revision != revision:
                        try:
                            return hf_hub_download(
                                repo_id=name_or_path,
                                filename=mxq_path,
                                revision=mxq_revision,
                            )
                        except EntryNotFoundError:
                            pass

                    cached = self._find_cached_mxq(name_or_path, mxq_path)
                    if cached is not None:
                        return cached
                    mxq_candidate = self._find_mxq_from_hub(
                        name_or_path,
                        mxq_path,
                        revision=mxq_revision or revision,
                    )
                    if mxq_candidate is None:
                        raise
                    return hf_hub_download(
                        repo_id=name_or_path,
                        filename=mxq_candidate,
                        revision=mxq_revision or revision,
                    )

        raise Exception(f"[Mobilint] Error: Could not locate {mxq_path}.")

    @staticmethod
    def _infer_hf_revision_from_cache(repo_id: str) -> Optional[str]:
        """Infers a HuggingFace Hub revision from the local cache.

        Searches the HF hub cache directory for the given repository and
        returns the first commit SHA found by inspecting the ``refs/`` and
        ``snapshots/`` directories.

        Args:
            repo_id: HuggingFace repository identifier in ``"owner/repo"``
                format.

        Returns:
            A commit SHA string if one is found in the local cache, or
            ``None`` if the cache cannot be located or read.
        """
        if not repo_id or "/" not in repo_id:
            return None

        cache_root = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
        if not cache_root:
            hf_home = os.getenv("HF_HOME") or os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
            )
            cache_root = os.path.join(hf_home, "hub")

        repo_dir = os.path.join(cache_root, f"models--{repo_id.replace('/', '--')}")
        refs_dir = os.path.join(repo_dir, "refs")
        if os.path.isdir(refs_dir):
            for ref_name in ("main", "master"):
                ref_path = os.path.join(refs_dir, ref_name)
                if os.path.isfile(ref_path):
                    try:
                        with open(ref_path, "r", encoding="utf-8") as f:
                            ref = f.read().strip()
                        if ref:
                            return ref
                    except OSError:
                        pass
            try:
                for entry in os.listdir(refs_dir):
                    ref_path = os.path.join(refs_dir, entry)
                    if os.path.isfile(ref_path):
                        with open(ref_path, "r", encoding="utf-8") as f:
                            ref = f.read().strip()
                        if ref:
                            return ref
            except OSError:
                pass

        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            try:
                for entry in os.listdir(snapshots_dir):
                    if os.path.isdir(os.path.join(snapshots_dir, entry)):
                        return entry
            except OSError:
                pass

        return None

    @staticmethod
    def _infer_revision_from_mxq_path(mxq_path: str) -> Optional[str]:
        basename = os.path.basename(mxq_path)
        stem, ext = os.path.splitext(basename)
        if ext != ".mxq" or "-" not in stem:
            return None

        revision = stem.rsplit("-", 1)[-1]
        if re.fullmatch(r"[A-Za-z]*\d[A-Za-z0-9]*", revision):
            return revision
        return None

    @staticmethod
    def _find_cached_mxq(repo_id: str, mxq_path: str) -> Optional[str]:
        """Searches the local HF hub cache for a cached MXQ file.

        Checks each snapshot directory for the given repo, looking first for
        an exact relative-path match and then for any file whose basename
        matches. Falls back to scanning the entire snapshot tree for any
        ``*.mxq`` file.

        Args:
            repo_id: HuggingFace repository identifier in ``"owner/repo"``
                format.
            mxq_path: Expected relative path or basename of the MXQ file
                within the repository.

        Returns:
            The absolute filesystem path to the cached file if found, or
            ``None`` otherwise.
        """
        if not repo_id or "/" not in repo_id:
            return None

        cache_root = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
        if not cache_root:
            hf_home = os.getenv("HF_HOME") or os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
            )
            cache_root = os.path.join(hf_home, "hub")

        repo_dir = os.path.join(cache_root, f"models--{repo_id.replace('/', '--')}")
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        rel_candidates = [mxq_path, os.path.basename(mxq_path)]
        try:
            for snapshot in os.listdir(snapshots_dir):
                snapshot_dir = os.path.join(snapshots_dir, snapshot)
                if not os.path.isdir(snapshot_dir):
                    continue
                for rel in rel_candidates:
                    candidate = os.path.join(snapshot_dir, rel)
                    if os.path.isfile(candidate):
                        return candidate
        except OSError:
            return None

        # Last resort: find any mxq in snapshots
        for root, _, files in os.walk(snapshots_dir):
            for name in files:
                if name.endswith(".mxq"):
                    return os.path.join(root, name)

        return None

    @staticmethod
    def _find_mxq_from_hub(repo_id: str, mxq_path: str, revision: Optional[str] = None) -> Optional[str]:
        try:
            files = HfApi().list_repo_files(repo_id=repo_id, revision=revision)
        except Exception:
            return None

        basename = os.path.basename(mxq_path)
        if basename in files:
            return basename
        if mxq_path in files:
            return mxq_path

        raise ValueError(f"Cannot find {mxq_path} file from HuggingFace repo: {repo_id}")

    # ---- Compatibility shims -------------------------------------------------

    @property
    def mxq_model(self) -> Optional["Model"]:
        """First-slot :class:`~qbruntime.Model` handle, or ``None`` before create().

        Preserved for callers written against the pre-multi-slot API
        (:mod:`cache_utils`, per-model utilities in ``modeling_utils``,
        etc.). New code that dispatches concurrent slots should read
        :attr:`mxq_models` directly.
        """
        return self.mxq_models[0] if self.mxq_models else None

    @property
    def acc(self) -> Optional["Accelerator"]:
        """First-inserted accelerator handle, or ``None`` before create()."""
        if not self.accs:
            return None
        return next(iter(self.accs.values()))

    # ---- Target helpers ------------------------------------------------------

    def _fallback_dev(self) -> int:
        """Return a single device index to prepend when migrating legacy target items."""
        dev = self._spec.dev_no_public()
        if isinstance(dev, list):
            return int(dev[0]) if dev else 0
        return int(dev)

    def _unique_devs_from_targets(self) -> List[int]:
        """Return the sorted set of device indices referenced by the canonical target lists.

        Falls back to :attr:`dev_no` sugar when both target lists are empty
        (defensive; :class:`NPUTargetSpec` normally guarantees at least one
        populated field).
        """
        return self._spec.unique_devices()

    def filter_cores_for(self, dev: int) -> List["CoreId"]:
        """Return the :class:`~qbruntime.CoreId` list for cores assigned to ``dev``.

        Reads :attr:`NPUTargetSpec.cores` and yields the entries whose
        device prefix matches ``dev``. Used to build a per-slot
        :class:`~qbruntime.ModelConfig` when the backend spans multiple
        devices.
        """
        result: List[CoreId] = []
        for s in self._spec.cores:
            parts = s.split(":")
            if len(parts) != 3:
                continue
            try:
                d_val, c_val, k_val = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if d_val != int(dev):
                continue
            try:
                result.append(CoreId(cluster_map[c_val], core_map[k_val]))
            except KeyError:
                logger.warning("Unknown cluster/core id in target_cores entry %r", s)
        return result

    def filter_clusters_for(self, dev: int) -> List["Cluster"]:
        """Return the :class:`~qbruntime.Cluster` list for clusters assigned to ``dev``.

        Reads :attr:`NPUTargetSpec.clusters` and yields the entries whose
        device prefix matches ``dev``. Used to build a per-slot
        :class:`~qbruntime.ModelConfig` for ``multi``/``global4``/``global8``
        modes.
        """
        result: List[Cluster] = []
        for s in self._spec.clusters:
            if not isinstance(s, str) or ":" not in s:
                continue
            try:
                d_val, c_val = int(s.split(":", 1)[0]), int(s.split(":", 1)[1])
            except ValueError:
                continue
            if d_val != int(dev):
                continue
            try:
                result.append(cluster_map[c_val])
            except KeyError:
                logger.warning("Unknown cluster id in target_clusters entry %r", s)
        return result

    # ---- Slot lifecycle ------------------------------------------------------

    def _make_slot_config(self, dev: int) -> "ModelConfig":
        """Build a :class:`~qbruntime.ModelConfig` restricted to ``dev``'s targets.

        Args:
            dev: Device index this slot is assigned to.

        Raises:
            ValueError: If ``self.core_mode`` is not one of the supported values.
            AssertionError: If ``"global8"`` mode is requested and ``dev`` does
                not carry both clusters.
        """
        mc = ModelConfig()
        if self.core_mode == "single":
            mc.set_single_core_mode(None, self.filter_cores_for(dev))
        elif self.core_mode == "multi":
            mc.set_multi_core_mode(self.filter_clusters_for(dev))
        elif self.core_mode == "global4":
            mc.set_global4_core_mode(self.filter_clusters_for(dev))
        elif self.core_mode == "global8":
            clusters = self.filter_clusters_for(dev)
            assert len(clusters) == 2, (
                f"core_mode='global8' requires both clusters on device {dev}; got {len(clusters)}."
            )
            mc.set_global8_core_mode()
        else:
            raise ValueError("core_mode must be single, multi, global4 or global8! value: " + str(self.core_mode))
        return mc

    @staticmethod
    def _probe_k_per_model(mxq_model: "Model") -> int:
        """Return the compiled batch axis ``K`` of ``mxq_model``, defaulting to ``1``.

        Reads :meth:`~qbruntime.Model.get_cache_infos` and returns the
        ``num_batches`` field of the first per-layer cache info entry.
        This is the authoritative K probe for LLM MXQs — the input shape
        of a batched LLM MXQ is ``(1, -1, hidden)``, so the leading input
        dimension cannot distinguish batched from non-batched artifacts.

        For MXQ artifacts without KV cache layers (e.g. vision models),
        :meth:`~qbruntime.Model.get_cache_infos` returns an empty list and
        the fallback of ``1`` is correct because there is no compiled
        batch axis to fan out along.
        """
        try:
            infos = mxq_model.get_cache_infos()
        except (AttributeError, QbRuntimeError) as exc:
            # Best-effort probe: any qbruntime failure here (BadAlloc or not)
            # is non-fatal — we fall back to K=1 rather than propagate,
            # because the caller can still run with a conservative slot count.
            logger.warning("Failed to probe k_per_model from get_cache_infos: %s", exc)
            return 1
        if not infos:
            return 1
        first = infos[0]
        try:
            k = int(getattr(first, "num_batches", 1) or 1)
        except (TypeError, ValueError):
            return 1
        return k if k > 0 else 1

    def _dispose_all_slots(self) -> None:
        """Dispose every previously-created Model, swallowing individual failures.

        The release path must not raise: callers use this both on normal
        teardown and on the rollback path after a partial ``create``/``launch``
        failure.
        """
        for m in self.mxq_models:
            try:
                m.dispose()
            except Exception as exc:  # noqa: BLE001 — release path must not raise.
                logger.warning("dispose failed during rollback: %s", exc)
        self.mxq_models = []
        self.model_dev_no = []

    def create(self) -> None:
        """Instantiate accelerators and load ``N`` MXQ Model slots.

        Groups the canonical target strings by device, opens one
        :class:`~qbruntime.Accelerator` per unique device, and loads
        ``N = ceil(max_batch_size / K)`` :class:`~qbruntime.Model` slots
        round-robin across those devices. ``K`` is probed from slot 0.
        The MXQ artifact is resolved via :meth:`check_model_path` exactly
        once; the resolved path is reused by every subsequent slot.

        On a device-memory ``BadAlloc`` (see :func:`_is_qbruntime_bad_alloc`),
        every previously loaded slot is disposed and the failure is rethrown
        as :class:`MobilintBackendAllocError` with slot / device / progress
        context. Any other :class:`~qbruntime.QbRuntimeError` (invalid MXQ
        artifact, incompatible target configuration, corrupted artifact,
        missing runtime dependency, ...) triggers the same partial-state
        rollback but is re-raised unchanged so the caller can distinguish
        a memory ceiling from a user-config or artifact bug.

        Raises:
            MobilintBackendAllocError: If any slot hits a device-memory
                ``BadAlloc``.
            QbRuntimeError: If any slot fails for a non-alloc reason (after
                partial-state rollback).
            ValueError: If ``self.core_mode`` is not one of the supported
                values.
            AssertionError: If ``"global8"`` mode is requested but a device
                does not cover both clusters.
        """
        unique_devs = self._unique_devs_from_targets()
        if not unique_devs:
            unique_devs = [self._fallback_dev()]

        self.accs = {int(d): Accelerator(int(d)) for d in unique_devs}
        self.mxq_models = []
        self.model_dev_no = []
        self.n_models = 0
        # Output layout is a fixed property of the compiled MXQ probed once
        # from slot 0. A dispose() + create() cycle may swap the artifact,
        # so invalidate any prior probe before loading new slots.
        self._output_layout_cached = None

        resolved_path: Optional[str] = None

        def _spawn_slot(slot_idx: int, dev: int) -> "Model":
            nonlocal resolved_path
            if resolved_path is None:
                resolved_path = self.check_model_path(self.mxq_path)
            mc = self._make_slot_config(dev)
            try:
                m = Model(resolved_path, mc)
            except QbRuntimeError as exc:
                succeeded = len(self.mxq_models)
                planned = max(self.n_models, slot_idx + 1)
                self._dispose_all_slots()
                self.accs = {}
                if _is_qbruntime_bad_alloc(exc):
                    raise MobilintBackendAllocError(
                        phase="create",
                        slot=slot_idx,
                        dev=int(dev),
                        succeeded_so_far=succeeded,
                        n_total=planned,
                        max_batch_size=self.max_batch_size,
                        k_per_model=self.k_per_model,
                        original=exc,
                    ) from exc
                # Non-alloc runtime failure (invalid MXQ, bad target config,
                # corrupted artifact, ...). Report progress on stderr so the
                # caller sees which slot broke, then re-raise unchanged.
                print(
                    f"[Mobilint] NPU backend create failed at slot {slot_idx} on device {dev} "
                    f"(succeeded {succeeded}/{planned}); re-raising qbruntime error unchanged.",
                    file=sys.stderr,
                )
                raise
            self.mxq_models.append(m)
            self.model_dev_no.append(int(dev))
            return m

        # Slot 0 tells us the compiled batch axis, which sets N for the
        # remaining slots.
        first_dev = int(unique_devs[0])
        first_model = _spawn_slot(0, first_dev)
        self.k_per_model = self._probe_k_per_model(first_model)
        self.n_models = max(1, math.ceil(self.max_batch_size / max(1, self.k_per_model)))

        for slot_idx in range(1, self.n_models):
            d = int(unique_devs[slot_idx % len(unique_devs)])
            _spawn_slot(slot_idx, d)

        # ``resolved_path`` is set by the first _spawn_slot call above.
        assert resolved_path is not None
        log_model_details(resolved_path, self)

    def launch(self) -> None:
        """Launch every loaded slot on its assigned accelerator.

        Must be called after :meth:`create`. On a device-memory ``BadAlloc``
        (see :func:`_is_qbruntime_bad_alloc`), every previously launched slot
        is disposed and the failure is rethrown as
        :class:`MobilintBackendAllocError`. Any other
        :class:`~qbruntime.QbRuntimeError` (invalid MXQ, bad target
        configuration, corrupted artifact, missing runtime dependency, ...)
        triggers the same partial-state rollback but is re-raised unchanged
        so the caller can distinguish a real memory ceiling from a
        user-config or artifact bug.

        Raises:
            MobilintBackendAllocError: If any slot hits a device-memory
                ``BadAlloc`` while launching.
            QbRuntimeError: If any slot fails to launch for a non-alloc
                reason (after partial-state rollback).
        """
        for i, m in enumerate(self.mxq_models):
            d = self.model_dev_no[i]
            try:
                m.launch(self.accs[d])
            except QbRuntimeError as exc:
                self._dispose_all_slots()
                self.accs = {}
                if _is_qbruntime_bad_alloc(exc):
                    raise MobilintBackendAllocError(
                        phase="launch",
                        slot=i,
                        dev=int(d),
                        succeeded_so_far=i,
                        n_total=self.n_models,
                        max_batch_size=self.max_batch_size,
                        k_per_model=self.k_per_model,
                        original=exc,
                    ) from exc
                # Non-alloc runtime failure — see :meth:`create` for the
                # rationale. Report progress on stderr then re-raise unchanged.
                print(
                    f"[Mobilint] NPU backend launch failed at slot {i} on device {d} "
                    f"(succeeded {i}/{self.n_models}); re-raising qbruntime error unchanged.",
                    file=sys.stderr,
                )
                raise

    def __call__(self, x):
        """Runs inference on slot 0.

        Preserved as a backward-compat shim for callers written against
        the single-slot API. New multi-slot callers should use
        :meth:`infer_slot` and manage cross-slot dispatch themselves
        because ``Model.infer`` is blocking.

        Args:
            x: Input data to pass to slot 0.

        Returns:
            The raw inference output produced by slot 0.
        """
        return self.mxq_models[0].infer(x)

    def infer_slot(self, i: int, x):
        """Runs inference on slot ``i``.

        Blocking. Callers that want to overlap slots must submit
        ``infer_slot`` calls from independent threads.

        Args:
            i: Slot index in ``[0, self.n_models)``.
            x: Input data to pass to that slot's Model.

        Returns:
            The raw inference output produced by slot ``i``.
        """
        return self.mxq_models[i].infer(x)

    def get_dtype(self) -> str:
        """Returns the input data type of slot 0 (identical across slots).

        Returns:
            A string representation of slot 0's input
            :class:`~qbruntime.DataType` (e.g. ``"DataType.Uint8"``).
        """
        return str(self.mxq_models[0].get_model_input_data_type())

    def get_input_buffer_info(self):
        """Returns the input buffer info of slot 0 (identical across slots).

        Returns:
            The ``get_input_buffer_info()`` return value produced by
            :class:`~qbruntime.Model` for slot 0.
        """
        return self.mxq_models[0].get_input_buffer_info()

    # ---- Multi-slot dispatch ------------------------------------------------

    @property
    def dispatcher(self):
        """Return the :class:`MultiSlotDispatcher` bound to this backend.

        Imported lazily so the base ``mblt_model_zoo.utils`` package does not
        pull the ``hf_transformers`` dependency graph into the compile-only
        code path.
        """
        if self._dispatcher is None:
            from ..hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher

            self._dispatcher = MultiSlotDispatcher(self)
        return self._dispatcher

    # ---- Output layout probe ------------------------------------------------

    @property
    def output_layout(self) -> Optional[Literal["n_items", "n_tokens"]]:
        """Return the compiled batched-infer output layout, or ``None`` when unknown.

        Two layouts show up in practice for a batched LLM MXQ call:
        ``"n_items"`` (one row per active batch item — a static last-token
        MXQ, or a dynamic-axis kernel that collapses the token axis for
        batched dispatch) and ``"n_tokens"`` (one row per input token,
        emitted by a truly dynamic-axis MXQ).

        The layout is a fixed property of the compiled MXQ — every slot in
        the backend runs the same artifact — so we probe it once from slot
        0's :meth:`qbruntime.Model.get_model_output_shape` output and cache
        the result. When the shape probe is ambiguous or the accessor is
        missing, this returns ``None`` and :class:`MultiSlotDispatcher`
        falls back to inspecting an unambiguous runtime group and pins the
        answer via :meth:`_set_output_layout` for the remainder of the
        process. Never defaults silently.
        """
        cached = self._output_layout_cached
        if cached is not None:
            return cached
        probed = self._probe_output_layout()
        if probed is not None:
            self._output_layout_cached = probed
        return probed

    def _set_output_layout(self, layout: Literal["n_items", "n_tokens"]) -> None:
        """Cache the runtime-observed output layout for the rest of this backend's life."""
        if layout not in ("n_items", "n_tokens"):
            raise ValueError(f"invalid output layout: {layout!r}")
        self._output_layout_cached = layout

    def _probe_output_layout(self) -> Optional[Literal["n_items", "n_tokens"]]:
        """Probe the batched output layout from slot 0's compiled shape.

        LLM MXQs declare their token axis at index ``-2`` of the first
        output shape. A ``-1`` sentinel marks the axis as dynamic (per-token
        streaming; layout is ``"n_tokens"``); any static value collapses
        the token axis to a single row per batch item (layout ``"n_items"``).

        A ``K > 1`` batched MXQ complicates the probe: the compiled batch
        axis can occupy position ``-2`` and be reported dynamic even though
        the runtime still emits per-item last-token logits (``"n_items"``).
        Shape metadata alone cannot distinguish "token axis dynamic" from
        "batch axis dynamic," so ``K > 1`` + dynamic ``-2`` returns ``None``
        and defers to the :class:`MultiSlotDispatcher` runtime fallback,
        which pins the answer from an unambiguous group.

        Returns ``None`` when the shape accessor is missing / errors, when
        the first output has fewer than two dims, or when the probe is
        ambiguous (see above) — the runtime fallback then pins the answer.
        """
        if not self.mxq_models:
            return None
        first = self.mxq_models[0]
        try:
            shapes = first.get_model_output_shape()
        except (AttributeError, QbRuntimeError) as exc:
            # Best-effort probe: any qbruntime failure here (BadAlloc or not)
            # is non-fatal — the dispatcher's runtime fallback pins the layout
            # from the first unambiguous group instead.
            logger.debug("output_layout: get_model_output_shape unavailable (%s)", exc)
            return None
        if not shapes:
            return None
        first_shape = tuple(shapes[0])
        if len(first_shape) < 2:
            return None
        try:
            token_axis = int(first_shape[-2])
        except (TypeError, ValueError):
            return None
        if token_axis != -1:
            return "n_items"
        if self.k_per_model > 1:
            # Ambiguous: the ``-1`` at position -2 could be the compiled batch
            # axis (K) rather than the token axis. Defer to the runtime
            # fallback rather than lock the wrong layout.
            return None
        return "n_tokens"

    def dispose(self) -> None:
        """Release every model and accelerator handle held by this backend.

        Safe to call multiple times.
        """
        self._dispose_all_slots()
        self.accs = {}

    @property
    def target_cores(self) -> List["CoreId"]:
        """Deserialize and return the list of target :class:`~qbruntime.CoreId` objects.

        Cores are stored internally on ``self._spec`` as canonical
        ``"d:c:k"`` strings. Entries whose device does not match
        ``self.dev_no`` are skipped so the existing single-device
        :meth:`create` path keeps its selection.

        When no explicit per-core list has been set, the getter falls back
        to expanding ``target_clusters`` into every core of each listed
        cluster on the current device. This preserves the historical
        ``target_clusters=[0, 1]`` short-hand for "use all 8 cores across
        both clusters" in ``single`` core mode without listing every core
        by hand.

        Returns:
            A list of :class:`~qbruntime.CoreId` objects representing the
            NPU cores selected on this device.
        """
        result: List["CoreId"] = []
        my_dev = self._fallback_dev()
        for s in self._spec.cores:
            try:
                parts = s.split(":")
                if len(parts) == 3:
                    d_val, c_val, r_val = int(parts[0]), int(parts[1]), int(parts[2])
                elif len(parts) == 2:
                    # Tolerate a stale legacy entry that slipped past normalization.
                    d_val, c_val, r_val = my_dev, int(parts[0]), int(parts[1])
                else:
                    raise ValueError(f"invalid entry: {s}")
                if d_val != my_dev:
                    continue
                result.append(CoreId(cluster_map[c_val], core_map[r_val]))
            except Exception as e:
                logger.warning("Target cores not serialized: %s", s)
                logger.warning("Error: %s", e)
        if result:
            return result

        # Fallback: expand target_clusters into their full 4-core set on this
        # device. Only kicks in when the caller left target_cores empty.
        for cluster_str in self._spec.clusters:
            try:
                if isinstance(cluster_str, str) and ":" in cluster_str:
                    d_val, c_val = int(cluster_str.split(":")[0]), int(cluster_str.split(":")[1])
                else:
                    d_val, c_val = my_dev, int(cluster_str)
                if d_val != my_dev:
                    continue
                cluster_enum = cluster_map[c_val]
            except Exception as e:
                logger.warning("Target cluster not serialized (fallback path): %s", cluster_str)
                logger.warning("Error: %s", e)
                continue
            for core_enum in (Core.Core0, Core.Core1, Core.Core2, Core.Core3):
                result.append(CoreId(cluster_enum, core_enum))
        return result

    @target_cores.setter
    def target_cores(self, values: List[Union[str, "CoreId"]]) -> None:
        """Atomically replace the target-cores component of ``self._spec``.

        Values pass through :meth:`NPUTargetSpec._with`, which forwards to
        :meth:`NPUTargetSpec.from_kwargs`. That path handles legacy
        migration (``CoreId`` / ``"c:k"``), grain fold/unfold, and the
        device-set consistency check in one atomic operation. Callers
        never observe the partial state where ``self._spec.cores`` has
        changed but ``self._spec.dev_no`` has not.
        """
        self._spec = self._spec._with(target_cores=list(values))

    @property
    def target_clusters(self) -> List["Cluster"]:
        """Deserialize and return the list of target :class:`~qbruntime.Cluster` objects.

        Clusters are stored internally on ``self._spec`` as canonical
        ``"d:c"`` strings. Entries whose device does not match
        ``self.dev_no`` are skipped so the existing single-device
        :meth:`create` path keeps its selection.

        Returns:
            A list of :class:`~qbruntime.Cluster` objects representing the
            NPU clusters selected on this device.
        """
        result: List["Cluster"] = []
        my_dev = self._fallback_dev()
        for s in self._spec.clusters:
            try:
                if isinstance(s, str) and ":" in s:
                    d_val, c_val = int(s.split(":")[0]), int(s.split(":")[1])
                else:
                    # Tolerate a stale legacy entry (bare int) that slipped past normalization.
                    d_val, c_val = my_dev, int(s)
                if d_val != my_dev:
                    continue
                result.append(cluster_map[c_val])
            except Exception as e:
                logger.warning("Target clusters not serialized: %s", s)
                logger.warning("Error: %s", e)
        return result

    @target_clusters.setter
    def target_clusters(self, values: List[Union[int, str, "Cluster"]]) -> None:
        """Atomically replace the target-clusters component of ``self._spec``.

        Values pass through :meth:`NPUTargetSpec._with`, which forwards to
        :meth:`NPUTargetSpec.from_kwargs`. That path handles legacy
        migration (``Cluster`` / ``int`` / bare ``"c"``), grain
        fold/unfold, and the device-set consistency check in one atomic
        operation.
        """
        self._spec = self._spec._with(target_clusters=list(values))

    def to_dict(self, prefix="") -> Dict[str, Any]:
        """Serializes the backend configuration to a flat dictionary.

        The canonical fully-qualified ``target_cores`` or ``target_clusters``
        list is passed through unchanged. Config-layer normalization
        (:meth:`NPUTargetSpec.from_kwargs`) is trusted to have already
        rewritten legacy inputs, so this method neither inspects nor
        rewrites the serialized entries.

        When canonical target strings are set, ``dev_no`` is derived from
        their device prefixes so the emitted dict round-trips through
        :meth:`NPUTargetSpec.from_kwargs` — the device-set consistency
        check requires ``dev_no`` and the target device set to agree once
        both are explicit. A single device collapses to an int; multiple
        devices emit a sorted list. When no targets are set (e.g. early
        construction before the config layer has expanded ``dev_no``
        sugar), the stored ``dev_no`` is passed through as-is.

        Args:
            prefix: Optional string to prepend to every key, useful when
                merging this configuration into a larger dictionary.

        Returns:
            A flat dictionary containing the serialized backend parameters.
        """
        p = prefix
        result: Dict[str, Any] = {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self._spec.dev_no_for_serialization(),
            f"{p}max_batch_size": self.max_batch_size,
            f"{p}core_mode": self.core_mode,
        }

        if self.core_mode == "single":
            result[f"{p}target_cores"] = list(self._spec.cores)
        else:
            result[f"{p}target_clusters"] = list(self._spec.clusters)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> "MobilintNPUBackend":
        """Constructs a :class:`MobilintNPUBackend` from a configuration dictionary.

        Trusts the config layer (:meth:`NPUTargetSpec.from_kwargs`) to have
        already rewritten ``target_cores`` / ``target_clusters`` entries
        into the canonical fully-qualified form. Keys are consumed from
        ``data`` and the instance is created with the extracted values.
        A warning is logged if both ``target_cores`` and ``target_clusters``
        keys are present, as only one is used depending on ``core_mode``.

        Args:
            data: A (possibly prefixed) flat dictionary produced by
                :meth:`to_dict` or a compatible configuration source.
                Keys are *popped* from this dictionary in place.
            prefix: The prefix that was used when the dictionary was
                serialized (must match the prefix used in :meth:`to_dict`).

        Returns:
            A new :class:`MobilintNPUBackend` instance configured from
            ``data``.
        """
        p = prefix
        if f"{p}target_cores" in data.keys() and f"{p}target_clusters" in data.keys():
            logger.warning("%starget_cores and %starget_clusters are both set!", p, p)
            logger.warning("If %score_mode is `single`, only %starget_cores will be used.", p, p)
            logger.warning(
                "If %score_mode is `multi`, `global4`, or `global8`, only %starget_clusters will be used.", p, p
            )

        return cls(
            name_or_path=data.pop("name_or_path", ""),
            mxq_path=data.pop(f"{p}mxq_path", ""),
            # ``None`` sentinel: distinguish "caller did not provide
            # dev_no" from "caller explicitly requested dev_no=0" so
            # :meth:`NPUTargetSpec.from_kwargs` skips its device-set
            # consistency check when the input dict lacks the key.
            dev_no=data.pop(f"{p}dev_no", None),
            max_batch_size=data.pop(f"{p}max_batch_size", 1),
            core_mode=data.pop(f"{p}core_mode", "single"),
            target_cores=data.pop(f"{p}target_cores", None),
            target_clusters=data.pop(f"{p}target_clusters", None),
            revision=data.pop(f"{p}revision", None),
            commit_hash=data.pop(f"{p}commit_hash", None),
        )
