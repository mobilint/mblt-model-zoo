from packaging.version import Version

try:
    from transformers import __version__ as transformers_version
except ImportError as exc:
    raise ImportError(
        "Mobilint Qwen3-ASR models require 'transformers>=4.57.0'. "
        'Please install or upgrade with: pip install -U "mblt_model_zoo[transformers]"'
    ) from exc

_MIN_TRANSFORMERS_VERSION = Version("4.57.0")


def _ensure_supported_transformers_version() -> None:
    """Raise an informative error when upstream Qwen3-ASR support is unavailable."""
    if Version(transformers_version) < _MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            "Mobilint Qwen3-ASR models require 'transformers>=4.57.0'. "
            f"Found transformers=={transformers_version}. "
            'Please upgrade transformers or reinstall with: pip install -U "mblt_model_zoo[transformers]"'
        )


_ensure_supported_transformers_version()

try:
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.configuration_qwen3_asr import (
        MobilintQwen3ASRConfig,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (
        MobilintQwen3ASRForConditionalGeneration,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_asr.processing_qwen3_asr import (
        MobilintQwen3ASRProcessor,
    )
except ImportError as exc:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        'Please run: pip install "mblt_model_zoo[transformers]"'
    ) from exc

__all__ = [
    "MobilintQwen3ASRConfig",
    "MobilintQwen3ASRForConditionalGeneration",
    "MobilintQwen3ASRProcessor",
]
