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

Backwards compatibility: for callers written against a single ``Model`` /
``Accelerator`` handle, :attr:`~MobilintNPUBackend.mxq_model` and
:attr:`~MobilintNPUBackend.acc` remain accessible and refer to the first slot.
"""

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig, QbRuntimeError

from .core_mode import CoreMode, normalize_core_mode
from .logging import log_model_details

logger = logging.getLogger(__name__)

cluster_map = {
    0: Cluster.Cluster0,
    1: Cluster.Cluster1,
}

core_map = {
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

# Default device index for ``dev_no`` when a caller does not pin one.
# Kept as a single named constant so the backend signature, ``from_dict``
# fallback, and config-layer normalizers stay in lock-step.
_DEFAULT_DEV_NO: int = 0


class MobilintBackendAllocError(RuntimeError):
    """Raised when a multi-slot backend fails to create or launch a slot.

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
        dev_no: Union[int, List[int]] = _DEFAULT_DEV_NO,
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
        self.dev_no = dev_no
        self.max_batch_size = max(1, max_batch_size)
        self.core_mode = normalize_core_mode(core_mode)

        # Multi-slot backing state; populated in create()/launch().
        # ``self.acc`` and ``self.mxq_model`` remain accessible as
        # compatibility properties that read the first slot.
        self.accs: Dict[int, "Accelerator"] = {}
        self.mxq_models: List["Model"] = []
        self.model_dev_no: List[int] = []
        self.k_per_model: int = 1
        self.n_models: int = 0

        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []

        self._target_clusters_serialized: List[str] = []
        self.target_clusters = target_clusters if target_clusters is not None else []

        # Snapshot of the canonical state produced by ``_normalize_npu_target_kwargs``
        # right after construction. ``_re_normalize_backend_state`` compares the
        # backend's current state against this snapshot to detect *which* fields
        # HF ``from_pretrained`` overrode via setattr — dev_no vs targets — so
        # the divergence resolution can trust the field the caller actually
        # changed instead of guessing. Populated by
        # :meth:`record_post_normalize_snapshot`; ``None`` until then.
        self._post_normalize_snapshot: Optional[Dict[str, Any]] = None

    def record_post_normalize_snapshot(self) -> None:
        """Capture the backend's canonical state right after initial normalization.

        Called by the config layer immediately after
        :func:`_normalize_npu_target_kwargs` and
        :meth:`MobilintNPUBackend.from_dict` produced a canonical state from the
        JSON payload. ``_re_normalize_backend_state`` diffs the backend's later
        state against this snapshot to identify which fields HF setter chains
        overrode.
        """
        self._post_normalize_snapshot = {
            "dev_no": self.dev_no,
            "core_mode": self.core_mode,
            "cores": tuple(self._target_cores_serialized or []),
            "clusters": tuple(self._target_clusters_serialized or []),
        }

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
        dev = getattr(self, "dev_no", 0)
        if isinstance(dev, (list, tuple)):
            return int(dev[0]) if dev else 0
        return int(dev)

    def _unique_devs_from_targets(self) -> List[int]:
        """Return the sorted set of device indices referenced by the canonical target lists.

        Falls back to :attr:`dev_no` sugar when both target lists are empty
        (e.g. before the config layer has run through
        ``_normalize_npu_target_kwargs``).
        """
        devs: set[int] = set()
        for s in self._target_cores_serialized or []:
            parts = s.split(":")
            if len(parts) == 3:
                try:
                    devs.add(int(parts[0]))
                except ValueError:
                    continue
        for s in self._target_clusters_serialized or []:
            if isinstance(s, str) and ":" in s:
                try:
                    devs.add(int(s.split(":", 1)[0]))
                except ValueError:
                    continue
        if devs:
            return sorted(devs)
        dev = getattr(self, "dev_no", 0)
        if isinstance(dev, (list, tuple)):
            return sorted({int(d) for d in dev}) or [0]
        return [int(dev)]

    def filter_cores_for(self, dev: int) -> List["CoreId"]:
        """Return the :class:`~qbruntime.CoreId` list for cores assigned to ``dev``.

        Reads :attr:`_target_cores_serialized` and yields the entries whose
        device prefix matches ``dev``. Used to build a per-slot
        :class:`~qbruntime.ModelConfig` when the backend spans multiple
        devices.
        """
        result: List[CoreId] = []
        for s in self._target_cores_serialized or []:
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

        Reads :attr:`_target_clusters_serialized` and yields the entries
        whose device prefix matches ``dev``. Used to build a per-slot
        :class:`~qbruntime.ModelConfig` for ``multi``/``global4``/``global8``
        modes.
        """
        result: List[Cluster] = []
        for s in self._target_clusters_serialized or []:
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

        On any :class:`~qbruntime.QbRuntimeError` (typically a device-memory
        ``BadAlloc``), every previously loaded slot is disposed and the
        failure is rethrown as :class:`MobilintBackendAllocError` with slot /
        device / progress context.

        Raises:
            MobilintBackendAllocError: If any slot fails to load.
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

        Must be called after :meth:`create`. On any
        :class:`~qbruntime.QbRuntimeError`, every previously launched slot
        is disposed and the failure is rethrown as
        :class:`MobilintBackendAllocError`.
        """
        for i, m in enumerate(self.mxq_models):
            d = self.model_dev_no[i]
            try:
                m.launch(self.accs[d])
            except QbRuntimeError as exc:
                self._dispose_all_slots()
                self.accs = {}
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

    def dispose(self) -> None:
        """Release every model and accelerator handle held by this backend.

        Safe to call multiple times.
        """
        self._dispose_all_slots()
        self.accs = {}

    @property
    def target_cores(self) -> List["CoreId"]:
        """Deserialize and return the list of target :class:`~qbruntime.CoreId` objects.

        Cores are stored internally as canonical ``"d:c:k"`` strings.
        Entries whose device does not match ``self.dev_no`` are skipped so
        the existing single-device ``create()`` path keeps its selection.

        When no explicit per-core list has been set, the getter falls back
        to expanding ``target_clusters`` into every core of each listed
        cluster. This preserves the historical ``target_clusters=[0, 1]``
        short-hand for "use all 8 cores across both clusters" in
        ``single`` core mode without listing every core by hand.

        Returns:
            A list of :class:`~qbruntime.CoreId` objects representing the
            NPU cores selected on this device.
        """
        result: List["CoreId"] = []
        serialized = getattr(self, "_target_cores_serialized", None) or []
        my_dev = self._fallback_dev()
        for s in serialized:
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
        cluster_serialized = getattr(self, "_target_clusters_serialized", None) or []
        for cluster_str in cluster_serialized:
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
    def target_cores(self, values: List[Union[str, "CoreId"]]):
        """Serialize and store the list of target cores.

        Accepts canonical fully-qualified ``"d:c:k"`` strings, legacy
        ``"c:k"`` strings, and :class:`~qbruntime.CoreId` objects. Legacy
        entries and ``CoreId`` objects are migrated to the canonical form
        using ``self.dev_no`` as the fallback device prefix.

        Args:
            values: A list of core identifiers.

        Raises:
            ValueError: If a string value has an unrecognized shape.
            TypeError: If a value is neither :class:`~qbruntime.CoreId`
                nor a string.
        """
        fallback_dev = self._fallback_dev()
        serialized = []
        for v in values:
            if isinstance(v, CoreId):
                serialized.append(f"{fallback_dev}:{cluster_to_int(v.cluster)}:{core_to_int(v.core)}")
            elif isinstance(v, str):
                n_colons = v.count(":")
                if n_colons == 2:
                    serialized.append(v)
                elif n_colons == 1:
                    serialized.append(f"{fallback_dev}:{v}")
                else:
                    raise ValueError(f"Invalid format: {v}")
            else:
                raise TypeError(f"Unsupported type: {type(v)}")

        self._target_cores_serialized = serialized

    @property
    def target_clusters(self) -> List["Cluster"]:
        """Deserialize and return the list of target :class:`~qbruntime.Cluster` objects.

        Clusters are stored internally as canonical ``"d:c"`` strings.
        Entries whose device does not match ``self.dev_no`` are skipped so
        the existing single-device ``create()`` path keeps its selection.

        Returns:
            A list of :class:`~qbruntime.Cluster` objects representing the
            NPU clusters selected on this device.
        """
        result = []
        serialized = getattr(self, "_target_clusters_serialized", None) or []
        my_dev = self._fallback_dev()
        for s in serialized:
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
    def target_clusters(self, values: List[Union[int, str, "Cluster"]]):
        """Serialize and store the list of target clusters.

        Accepts canonical fully-qualified ``"d:c"`` strings, legacy bare
        ``"c"`` strings, integer cluster indices, and
        :class:`~qbruntime.Cluster` objects. Legacy entries are migrated
        to the canonical form using ``self.dev_no`` as the fallback
        device prefix.

        Args:
            values: A list of cluster identifiers.

        Raises:
            ValueError: If a string value has an unrecognized shape.
            TypeError: If a value is not one of the accepted types.
        """
        fallback_dev = self._fallback_dev()
        serialized = []
        for v in values:
            if isinstance(v, Cluster):
                serialized.append(f"{fallback_dev}:{cluster_to_int(v)}")
            elif isinstance(v, bool):
                raise TypeError(f"Unsupported type: {type(v)}")
            elif isinstance(v, int):
                serialized.append(f"{fallback_dev}:{v}")
            elif isinstance(v, str):
                n_colons = v.count(":")
                if n_colons == 1:
                    serialized.append(v)
                elif n_colons == 0 and v.isdigit():
                    serialized.append(f"{fallback_dev}:{v}")
                else:
                    raise ValueError(f"Invalid format: {v}")
            else:
                raise TypeError(f"Unsupported type: {type(v)}")

        self._target_clusters_serialized = serialized

    def to_dict(self, prefix="") -> Dict[str, Any]:
        """Serializes the backend configuration to a flat dictionary.

        The canonical fully-qualified ``target_cores`` or ``target_clusters``
        list is passed through unchanged. Config-layer normalization
        (``_normalize_npu_target_kwargs``) is trusted to have already
        rewritten legacy inputs, so this method neither inspects nor
        rewrites the serialized entries.

        When canonical target strings are set, ``dev_no`` is derived from
        their device prefixes so the emitted dict round-trips through
        ``_normalize_npu_target_kwargs`` — the config-layer device-set
        consistency check requires ``dev_no`` and the target device set to
        agree once both are explicit. A single device collapses to an int;
        multiple devices emit a sorted list. When no targets are set (e.g.
        early construction before the config layer has expanded ``dev_no``
        sugar), the stored ``self.dev_no`` is passed through as-is.

        Args:
            prefix: Optional string to prepend to every key, useful when
                merging this configuration into a larger dictionary.

        Returns:
            A flat dictionary containing the serialized backend parameters.
        """
        p = prefix
        result = {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self._dev_no_for_serialization(),
            f"{p}max_batch_size": self.max_batch_size,
            f"{p}core_mode": self.core_mode,
        }

        if self.core_mode == "single":
            result[f"{p}target_cores"] = self._target_cores_serialized
        else:
            result[f"{p}target_clusters"] = self._target_clusters_serialized

        return result

    def _dev_no_for_serialization(self) -> Union[int, List[int]]:
        """Return ``dev_no`` derived from canonical targets, else the stored value.

        Parsing failures on the target strings fall back to the stored
        ``self.dev_no`` rather than silently dropping the field.
        """
        if self._target_cores_serialized:
            source = self._target_cores_serialized
        elif self._target_clusters_serialized:
            source = self._target_clusters_serialized
        else:
            return self.dev_no

        devs: set[int] = set()
        for s in source:
            try:
                devs.add(int(s.split(":", 1)[0]))
            except (ValueError, AttributeError, IndexError):
                return self.dev_no

        if not devs:
            return self.dev_no

        sorted_devs = sorted(devs)
        return sorted_devs if len(sorted_devs) > 1 else sorted_devs[0]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> "MobilintNPUBackend":
        """Constructs a :class:`MobilintNPUBackend` from a configuration dictionary.

        Trusts the config layer (``_normalize_npu_target_kwargs``) to have
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
            dev_no=data.pop(f"{p}dev_no", _DEFAULT_DEV_NO),
            max_batch_size=data.pop(f"{p}max_batch_size", 1),
            core_mode=data.pop(f"{p}core_mode", "single"),
            target_cores=data.pop(f"{p}target_cores", None),
            target_clusters=data.pop(f"{p}target_clusters", None),
            revision=data.pop(f"{p}revision", None),
            commit_hash=data.pop(f"{p}commit_hash", None),
        )
