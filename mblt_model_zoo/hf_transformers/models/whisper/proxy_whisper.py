"""Proxy exports for the Mobilint Whisper Hub repository."""

from __future__ import annotations

from typing import Any

from transformers import WhisperProcessor, WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor as HFWhisperFeatureExtractor,
)

try:
    from mblt_model_zoo.hf_transformers.models.whisper.configuration_whisper import (
        MobilintWhisperConfig,
    )
    from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
        MobilintWhisperForConditionalGeneration,
    )
except ImportError as exc:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    ) from exc


class WhisperFeatureExtractor(HFWhisperFeatureExtractor):
    """Feature extractor shim for legacy Whisper timestamp preprocessing."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build input features and backfill `num_frames` for timestamp decoding.

        Args:
            *args: Positional arguments forwarded to `WhisperFeatureExtractor.__call__`.
            **kwargs: Keyword arguments forwarded to `WhisperFeatureExtractor.__call__`.

        Returns:
            The batch feature output from the parent feature extractor.
        """
        processed = super().__call__(*args, **kwargs)

        if (
            kwargs.get("return_token_timestamps")
            and "num_frames" not in processed
            and "attention_mask" in processed
        ):
            processed["num_frames"] = processed["attention_mask"].sum(-1)

        return processed


__all__ = [
    "MobilintWhisperConfig",
    "MobilintWhisperForConditionalGeneration",
    "WhisperFeatureExtractor",
    "WhisperProcessor",
    "WhisperTokenizer",
]
