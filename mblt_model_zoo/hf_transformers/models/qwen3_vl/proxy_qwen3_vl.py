"""Remote-code proxy exports for Mobilint Qwen3-VL models."""

from packaging.version import Version

try:
    from transformers import __version__ as transformers_version
except ImportError as exc:
    raise ImportError(
        "Mobilint Qwen3-VL models require 'transformers>=4.57.0'. "
        'Please install or upgrade with: pip install -U "mblt_model_zoo[transformers]"'
    ) from exc

_MIN_TRANSFORMERS_VERSION = Version("4.57.0")


def _ensure_supported_transformers_version() -> None:
    """Raise an informative error when upstream Qwen3-VL support is unavailable."""
    if Version(transformers_version) < _MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            "Mobilint Qwen3-VL models require 'transformers>=4.57.0'. "
            f"Found transformers=={transformers_version}. "
            'Please upgrade transformers or reinstall with: pip install -U "mblt_model_zoo[transformers]"'
        )


_ensure_supported_transformers_version()

try:
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (
        MobilintQwen3VLConfig,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
        MobilintQwen3VLForConditionalGeneration,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
        MobilintQwen3VLProcessor,
    )
except ImportError as exc:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        'Please run: pip install "mblt_model_zoo[transformers]"'
    ) from exc

__all__ = ["MobilintQwen3VLConfig", "MobilintQwen3VLForConditionalGeneration", "MobilintQwen3VLProcessor"]
