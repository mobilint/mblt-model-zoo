import hashlib
import os

_VERBOSE_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_verbose_enabled() -> bool:
    return os.getenv("MBLT_MODEL_ZOO_VERBOSE", "").lower() in _VERBOSE_TRUE_VALUES


def _md5_hash_from_file(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def log_model_details(model_path: str) -> None:
    """Print model metadata when verbose logging is enabled."""
    if not _is_verbose_enabled():
        return

    print("Model Initialized")
    print(f"Model Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"Model Hash: {_md5_hash_from_file(model_path)}")
