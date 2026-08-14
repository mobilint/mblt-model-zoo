"""NPU backend implementation for Mobilint hardware accelerators.

Provides the :class:`MobilintNPUBackend` class which wraps the ``qbruntime``
library to load, configure, and run MXQ models on Mobilint NPU devices.
It also handles model-file resolution, including downloading artifacts from
HuggingFace Hub when a local path is not found.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig

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


class MobilintNPUBackend:
    """Backend that runs MXQ models on the Mobilint NPU.

    Wraps the ``qbruntime`` ``Model`` and ``Accelerator`` APIs and provides
    helpers for locating MXQ model files either locally or on HuggingFace Hub.

    Class Attributes:
        num_of_clusters: Total number of hardware clusters available.
        num_of_cores_in_cluster: Number of cores per cluster.
    """

    num_of_clusters = 2
    num_of_cores_in_cluster = 4

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: Union[int, List[int]] = 0,
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

        # Declared here; set during create()
        self.acc: Optional["Accelerator"] = None
        self.mxq_model: Optional["Model"] = None

        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []

        self._target_clusters_serialized: List[str] = []
        self.target_clusters = target_clusters if target_clusters is not None else []

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

    def create(self):
        """Creates and configures the NPU accelerator and loads the model.

        Instantiates a :class:`~qbruntime.Accelerator` for ``self.dev_no``,
        builds a :class:`~qbruntime.ModelConfig` according to ``self.core_mode``
        and the selected targets, resolves ``self.mxq_path`` via
        :meth:`check_model_path`, and loads the MXQ model.

        Raises:
            ValueError: If ``self.core_mode`` is not one of the supported
                values.
            AssertionError: If ``"global8"`` mode is requested but fewer than
                two clusters are specified.
        """
        self.acc = Accelerator(self.dev_no)
        mc = ModelConfig()

        if self.core_mode == "single":
            mc.set_single_core_mode(None, self.target_cores)
        elif self.core_mode == "multi":
            mc.set_multi_core_mode(self.target_clusters)
        elif self.core_mode == "global4":
            mc.set_global4_core_mode(self.target_clusters)
        elif self.core_mode == "global8":
            assert len(self.target_clusters) == 2, "global8 must contain every cores!"
            mc.set_global8_core_mode()
        else:
            raise ValueError("core_mode must be single, multi, global4 or global8! value: " + self.core_mode)

        model_path = self.check_model_path(self.mxq_path)
        self.mxq_model = Model(model_path, mc)
        log_model_details(model_path, self)

    def launch(self):
        """Launches the loaded MXQ model on the accelerator.

        Must be called after :meth:`create` before performing inference.
        """
        self.mxq_model.launch(self.acc)

    def __call__(self, x):
        """Runs inference on the NPU model.

        Args:
            x: Input data to pass to the model.

        Returns:
            The raw inference output produced by the MXQ model.
        """
        return self.mxq_model.infer(x)

    def get_dtype(self):
        """Returns the input data type of the loaded model.

        Returns:
            A string representation of the model's input
            :class:`~qbruntime.DataType` (e.g. ``"DataType.Uint8"``).
        """
        return str(self.mxq_model.get_model_input_data_type())

    def dispose(self):
        """Releases hardware resources held by the model.

        Should be called when inference is complete to free NPU memory and
        any associated accelerator state.
        """
        self.mxq_model.dispose()

    def _fallback_dev(self) -> int:
        """Return a single device index to prepend when migrating legacy target items."""
        dev = getattr(self, "dev_no", 0)
        if isinstance(dev, (list, tuple)):
            return int(dev[0]) if dev else 0
        return int(dev)

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
                serialized.append(f"{fallback_dev}:{v.cluster.value}:{v.core.value}")
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
                serialized.append(f"{fallback_dev}:{v.value}")
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

        The ``target_cores`` or ``target_clusters`` key is included depending
        on ``core_mode``.

        Args:
            prefix: Optional string to prepend to every key, useful when
                merging this configuration into a larger dictionary.

        Returns:
            A flat dictionary containing the serialized backend parameters.
        """
        p = prefix
        result = {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self.dev_no,
            f"{p}max_batch_size": self.max_batch_size,
            f"{p}core_mode": self.core_mode,
        }

        if self.core_mode == "single":
            result[f"{p}target_cores"] = self._target_cores_serialized
        else:
            result[f"{p}target_clusters"] = self._target_clusters_serialized

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> "MobilintNPUBackend":
        """Constructs a :class:`MobilintNPUBackend` from a configuration dictionary.

        Keys are consumed from ``data`` and the instance is created with the
        extracted values. A warning is logged if both ``target_cores`` and
        ``target_clusters`` keys are present, as only one is used depending
        on ``core_mode``.

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
            dev_no=data.pop(f"{p}dev_no", 0),
            max_batch_size=data.pop(f"{p}max_batch_size", 1),
            core_mode=data.pop(f"{p}core_mode", "single"),
            target_cores=data.pop(f"{p}target_cores", None),
            target_clusters=data.pop(f"{p}target_clusters", None),
            revision=data.pop(f"{p}revision", None),
            commit_hash=data.pop(f"{p}commit_hash", None),
        )
