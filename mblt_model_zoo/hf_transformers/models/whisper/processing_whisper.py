"""Processing utilities for Mobilint Whisper models."""

from __future__ import annotations

from typing import Any

from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor as HFWhisperFeatureExtractor,
)


def _sum_attention_mask_frames(attention_mask: Any) -> Any:
    """Sum an attention mask over the frame axis.

    Args:
        attention_mask: Attention mask returned by the parent feature extractor.

    Returns:
        Per-sample frame counts.

    Raises:
        TypeError: If the mask does not support frame-wise summation.
    """
    if hasattr(attention_mask, "sum"):
        return attention_mask.sum(-1)

    if isinstance(attention_mask, list):
        if not attention_mask:
            return []
        if all(isinstance(row, list) for row in attention_mask):
            return [sum(row) for row in attention_mask]
        return sum(attention_mask)

    raise TypeError(f"Unsupported attention_mask type for timestamp frame counting: {type(attention_mask).__name__}")


class MobilintWhisperFeatureExtractor(HFWhisperFeatureExtractor):
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
            processed["num_frames"] = _sum_attention_mask_frames(processed["attention_mask"])

        return processed


__all__ = ["MobilintWhisperFeatureExtractor"]
