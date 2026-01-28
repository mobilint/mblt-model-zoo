import os
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig
from transformers.utils.generic import logging
from transformers.modeling_utils import PreTrainedModel

from ...utils.logging import log_model_details

logger = logging.get_logger(__name__)

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

class PretrainedOnlyMixin(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        _internal_call = kwargs.pop("_internal_call", False)
        
        if not _internal_call:
            cls_name = self.__class__.__name__
            raise RuntimeError(
                f"Direct instantiation of {cls_name} is not allowed.\n"
                f"Please use `{cls_name}.from_pretrained(...)` to load the NPU model correctly."
            )
            
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        kwargs["_internal_call"] = True
        embedding_weight_path = kwargs.pop("embedding_weight", None)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if embedding_weight_path:
            cls._inject_custom_embeddings(model, embedding_weight_path)

        return model

    @staticmethod
    def _inject_custom_embeddings(model: PreTrainedModel, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom embedding path not found: {path}")

        print(f"[Mobilint] Loading custom embeddings from: {path}")
        
        custom_data = torch.load(path, map_location="cpu")
        
        # Handle dict (state_dict) vs Tensor
        if isinstance(custom_data, dict):
            # Try to find common keys for weights
            if "weight" in custom_data:
                new_weight = custom_data["weight"]
            else:
                # If ambiguous, take the first value
                new_weight = next(iter(custom_data.values()))
        elif isinstance(custom_data, torch.Tensor):
            new_weight = custom_data
        else:
            raise ValueError(f"Unsupported data format in {path}. Expected dict or Tensor.")

        input_embeddings = model.get_input_embeddings()
        
        original_vocab_size = input_embeddings.weight.shape[0]
        new_vocab_size = new_weight.shape[0]
        embed_dim = input_embeddings.weight.shape[1]

        if new_weight.shape[1] != embed_dim:
            raise ValueError(f"Embedding dimension mismatch! Model expects {embed_dim}, but file has {new_weight.shape[1]}")

        if original_vocab_size != new_vocab_size:
            raise ValueError(f"Vocab size mismatch! Model expects {original_vocab_size}, but file has {new_vocab_size}")

        with torch.no_grad():
            input_embeddings.weight.data = new_weight.to(
                device=input_embeddings.weight.device,
                dtype=input_embeddings.weight.dtype
            )
        
        print("[Mobilint] Custom embeddings successfully injected.")


class MobilintNPUBackend:
    num_of_clusters = 2
    num_of_cores_in_cluster = 4

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, 'CoreId']]] = None,
        **kwargs
    ):
        self.name_or_path: str = "" # will be populated in MobilintModelMixin
        self.mxq_path = mxq_path
        self.dev_no = dev_no
        self.core_mode = core_mode
        
        self._target_cores_serialized: List[str] = []
        self.target_cores = target_cores if target_cores is not None else []
        
    def check_model_path(self, mxq_path: str) -> str:
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
            revision = (
                getattr(self, "revision", None)
                or getattr(self, "_commit_hash", None)
                or self._infer_hf_revision_from_cache(self.name_or_path)
            )
            try:
                return hf_hub_download(
                    repo_id=self.name_or_path,
                    filename=mxq_path,
                    revision=revision,
                )
            except EntryNotFoundError:
                try:
                    return hf_hub_download(
                        repo_id=self.name_or_path,
                        filename=mxq_path,
                    )
                except EntryNotFoundError:
                    cached = self._find_cached_mxq(self.name_or_path, mxq_path)
                    if cached is not None:
                        return cached
                    mxq_candidate = self._find_mxq_from_hub(self.name_or_path, mxq_path)
                    if mxq_candidate is None:
                        raise
                    return hf_hub_download(
                        repo_id=self.name_or_path,
                        filename=mxq_candidate,
                        revision=revision,
                    )
            
        raise Exception(f"[Mobilint] Error: Could not locate {mxq_path}.")

    @staticmethod
    def _infer_hf_revision_from_cache(repo_id: str) -> Optional[str]:
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
        try:
            files = HfApi().list_repo_files(repo_id=repo_id)
        except Exception:
            return None

        basename = os.path.basename(mxq_path)
        if basename in files:
            return basename
        if mxq_path in files:
            return mxq_path

        mxq_files = [f for f in files if f.endswith(".mxq")]
        if not mxq_files:
            return None

        for f in mxq_files:
            if os.path.basename(f) == basename:
                return f

        base_stem = os.path.splitext(basename)[0]
        for f in mxq_files:
            if base_stem and base_stem in os.path.basename(f):
                return f

        return mxq_files[0]
    
    def launch(self):
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
        log_model_details(model_path)
        self.mxq_model.launch(self.acc)

    def dispose(self):
        self.mxq_model.dispose()

    @property
    def target_cores(self) -> List['CoreId']:
        result = []
        if not hasattr(self, "_target_cores_serialized"):
            return []
            
        for s in self._target_cores_serialized:
            try:
                c_val, r_val = map(int, s.split(':'))
                result.append(CoreId(cluster_map[c_val], core_map[r_val]))
            except Exception as e:
                logger.warning("Target cores not serialized: %s" % s)
                logger.warning("Error: %s" % e)
                pass
        return result

    @target_cores.setter
    def target_cores(self, values: List[Union[str, 'CoreId']]):
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
    def target_clusters(self) -> List['Cluster']:        
        core_id_lists = [[core_id for core_id in self.target_cores if core_id.cluster == cluster_map[i]] for i in range(self.num_of_clusters)]
                
        for core_ids in core_id_lists:
            if len(core_ids) != self.num_of_cores_in_cluster and len(core_ids) != 0:
                raise ValueError(f"Target cores must include every cores in a cluster! core_ids: {self._target_cores_serialized}")
        
        return [core_ids[0].cluster for core_ids in core_id_lists if len(core_ids) == self.num_of_cores_in_cluster]

    @target_cores.setter
    def target_clusters(self, values: List[Union[int, 'Cluster']]):
        clusters = []
        for v in values:
            if isinstance(v, Cluster):
                clusters.append(v)
            elif isinstance(v, int):
                clusters.append(cluster_map[v])
            else:
                raise TypeError(f"Unsupported type: {type(v)}")
        
        core_ids = []
        for cluster in clusters:
            core_ids = [CoreId(cluster, core) for core in core_map.values()]

        self.target_cores = core_ids

    def to_dict(self, prefix="") -> Dict[str, Any]:
        p = prefix
        return {
            f"{p}mxq_path": self.mxq_path,
            f"{p}dev_no": self.dev_no,
            f"{p}core_mode": self.core_mode,
            f"{p}target_cores": self._target_cores_serialized
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], prefix: str = "") -> 'MobilintNPUBackend':
        p = prefix
        if f"{p}target_cores" in data.keys() and f"{p}target_clusters" in data.keys():
            logger.warning(f"{p}target_cores and {p}target_clusters are both set! {p}target_clusters will take priority")

        return cls(
            name_or_path=data.pop("name_or_path", ""),
            mxq_path=data.pop(f"{p}mxq_path", ""),
            dev_no=data.pop(f"{p}dev_no", 0),
            core_mode=data.pop(f"{p}core_mode", "single"),
            target_cores=data.pop(f"{p}target_cores", None),
            target_clusters=data.pop(f"{p}target_clusters", None)
        )
