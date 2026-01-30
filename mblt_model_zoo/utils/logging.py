import hashlib
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .npu_backend import MobilintNPUBackend

_VERBOSE_TRUE_VALUES = {"1", "true", "yes", "on"}

def _is_verbose_enabled() -> bool:
    return os.getenv("MBLT_MODEL_ZOO_VERBOSE", "").lower() in _VERBOSE_TRUE_VALUES


def _md5_hash_from_file(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def log_model_details(model_path: str, npu_backend: Optional["MobilintNPUBackend"] = None) -> None:
    """Print model metadata when verbose logging is enabled."""
    if not _is_verbose_enabled():
        return

    print("Model Initialized")
    print(f"Model Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"Model Hash: {_md5_hash_from_file(model_path)}")

    if npu_backend is not None:
        print(f"Device Number: {npu_backend.dev_no}")
        print(f"Core Mode: {npu_backend.core_mode}")
        if npu_backend.core_mode == "single":
            print(f"Target Cores: {npu_backend.target_cores}")
        else:
            print(f"Target Clusters: {npu_backend.target_clusters}")
        if npu_backend.mxq_model.get_num_model_variants() == 1:
            print(f"Model Input Shape: {npu_backend.mxq_model.get_model_input_shape()}")
            print(f"Model Output Shape: {npu_backend.mxq_model.get_model_output_shape()}")
        else:
            for i in range(npu_backend.mxq_model.get_num_model_variants()):
                print(f"Model Variant {i}")
                print(f"\tInput Shape: {npu_backend.mxq_model.get_model_variant_handle(i).get_model_input_shape()}")
                print(f"\tOutput Shape: {npu_backend.mxq_model.get_model_variant_handle(i).get_model_output_shape()}")