"""NPU backend implementation for Mobilint hardware accelerators.

Provides the :class:`MobilintNPUBackend` class which wraps the ``qbruntime``
library to load, configure, and run MXQ models on Mobilint NPU devices.
It also handles model-file resolution, including downloading artifacts from
HuggingFace Hub when a local path is not found.
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig

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
        dev_no: int = 0,
        max_batch_size: int = 1,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[List[Union[int, "Cluster"]]] = None,
        revision: Optional[str] = None,
        commit_hash: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the NPU backend configuration.

        Args:
            mxq_path: Path to the compiled MXQ model file.
            dev_no: Accelerator device number to use.
            core_mode: Execution mode that determines how NPU cores are
                allocated. One of ``"single"``, ``"multi"``, ``"global4"``,
                or ``"global8"``.
            target_cores: List of core identifiers (as ``"cluster:core"``
                strings or :class:`~qbruntime.CoreId` objects) used in
                ``"single"`` mode. ``None`` means all cores.
            target_clusters: List of cluster identifiers (as integers or
                :class:`~qbruntime.Cluster` objects) used in ``"multi"``,
                ``"global4"``, and ``"global8"`` modes.
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
        self.core_mode = core_mode

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
                    cached = self._find_cached_mxq(name_or_path, mxq_path)
                    if cached is not None:
                        return cached
                    mxq_candidate = self._find_mxq_from_hub(name_or_path, mxq_path)
                    if mxq_candidate is None:
                        raise
                    return hf_hub_download(
                        repo_id=name_or_path,
                        filename=mxq_candidate,
                        revision=revision,
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
    def _find_mxq_from_hub(repo_id: str, mxq_path: str) -> Optional[str]:
        """Finds the best-matching MXQ filename in a HuggingFace repository.

        Queries the Hub file listing and checks whether the exact path or
        the basename of ``mxq_path`` is present.

        Args:
            repo_id: HuggingFace repository identifier in ``"owner/repo"``
                format.
            mxq_path: Expected relative path or basename of the MXQ file
                within the repository.

        Returns:
            The matching filename as it appears in the repository, or ``None``
            if the repository listing cannot be retrieved.

        Raises:
            ValueError: If the repository is reachable but neither the full
                path nor the basename of ``mxq_path`` is present.
        """
        try:
            files = HfApi().list_repo_files(repo_id=repo_id)
        except Exception:
            return None

        basename = os.path.basename(mxq_path)
        if basename in files:
            return basename
        if mxq_path in files:
            return mxq_path

        raise ValueError(f"Cannot find {mxq_path} file from HuggingFace repo: f{repo_id}")

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

    @property
    def target_cores(self) -> List["CoreId"]:
        """Deserializes and returns the list of target :class:`~qbruntime.CoreId` objects.

        Cores are stored internally as ``"cluster:core"`` strings and
        converted to :class:`~qbruntime.CoreId` instances on access.

        Returns:
            A list of :class:`~qbruntime.CoreId` objects representing the
            configured NPU cores.
        """
        result = []
        if not hasattr(self, "_target_cores_serialized"):
            return []

        for s in self._target_cores_serialized:
            try:
                c_val, r_val = map(int, s.split(":"))
                result.append(CoreId(cluster_map[c_val], core_map[r_val]))
            except Exception as e:
                logger.warning("Target cores not serialized: %s", s)
                logger.warning("Error: %s", e)
        return result

    @target_cores.setter
    def target_cores(self, values: List[Union[str, "CoreId"]]):
        """Serializes and stores the list of target cores.

        Args:
            values: A list of core identifiers, either as
                :class:`~qbruntime.CoreId` objects or ``"cluster:core"``
                formatted strings.

        Raises:
            ValueError: If a string value does not contain ``":"``.
            TypeError: If a value is neither a :class:`~qbruntime.CoreId`
                nor a string.
        """
        serialized = []
        for v in values:
            if isinstance(v, CoreId):
                serialized.append(f"{v.cluster.value}:{v.core.value}")
            elif isinstance(v, str):
                if ":" in v:
                    serialized.append(v)
                else:
                    raise ValueError(f"Invalid format: {v}")
            else:
                raise TypeError(f"Unsupported type: {type(v)}")

        self._target_cores_serialized = serialized

    @property
    def target_clusters(self) -> List["Cluster"]:
        """Deserializes and returns the list of target :class:`~qbruntime.Cluster` objects.

        Clusters are stored internally as integer strings and converted to
        :class:`~qbruntime.Cluster` instances on access.

        Returns:
            A list of :class:`~qbruntime.Cluster` objects representing the
            configured NPU clusters.
        """
        result = []
        if not hasattr(self, "_target_clusters_serialized"):
            return []

        for s in self._target_clusters_serialized:
            try:
                c_val = int(s)
                result.append(cluster_map[c_val])
            except Exception as e:
                logger.warning("Target clusters not serialized: %s", s)
                logger.warning("Error: %s", e)
        return result

    @target_clusters.setter
    def target_clusters(self, values: List[Union[int, "Cluster"]]):
        """Serializes and stores the list of target clusters.

        Args:
            values: A list of cluster identifiers, either as
                :class:`~qbruntime.Cluster` objects or integer indices.

        Raises:
            TypeError: If a value is neither a :class:`~qbruntime.Cluster`
                nor an integer.
        """
        serialized = []
        for v in values:
            if isinstance(v, Cluster):
                serialized.append(v.value)
            elif isinstance(v, int):
                serialized.append(v)
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
