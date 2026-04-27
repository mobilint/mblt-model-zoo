"""Proxy exports for the Mobilint Whisper Hub repository."""

from transformers import WhisperProcessor, WhisperTokenizer

try:
    from mblt_model_zoo.hf_transformers.models.whisper.configuration_whisper import (
        MobilintWhisperConfig,
    )
    from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
        MobilintWhisperForConditionalGeneration,
    )
    from mblt_model_zoo.hf_transformers.models.whisper.processing_whisper import (
        MobilintWhisperFeatureExtractor,
    )
except ImportError as exc:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    ) from exc


__all__ = [
    "MobilintWhisperFeatureExtractor",
    "MobilintWhisperConfig",
    "MobilintWhisperForConditionalGeneration",
    "WhisperProcessor",
    "WhisperTokenizer",
]
