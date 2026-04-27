"""Regression tests for the Whisper feature extractor proxy contract."""

import pytest
import torch
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor as HFWhisperFeatureExtractor,
)

from mblt_model_zoo.hf_transformers.models.whisper import proxy_whisper
from mblt_model_zoo.hf_transformers.models.whisper.processing_whisper import MobilintWhisperFeatureExtractor


def test_whisper_feature_extractor_counts_list_attention_mask_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute timestamp frame counts when attention masks are Python lists."""
    monkeypatch.setattr(
        HFWhisperFeatureExtractor,
        "__call__",
        lambda *args, **kwargs: {"attention_mask": [[1, 1, 0], [1, 0, 0]]},
    )
    extractor = object.__new__(MobilintWhisperFeatureExtractor)

    processed = MobilintWhisperFeatureExtractor.__call__(
        extractor,
        [0.0, 0.0],
        return_token_timestamps=True,
    )

    assert processed["num_frames"] == [2, 1]


def test_whisper_feature_extractor_keeps_tensor_attention_mask_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the native tensor sum path for tensorized attention masks."""
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    monkeypatch.setattr(
        HFWhisperFeatureExtractor,
        "__call__",
        lambda *args, **kwargs: {"attention_mask": attention_mask},
    )
    extractor = object.__new__(MobilintWhisperFeatureExtractor)

    processed = MobilintWhisperFeatureExtractor.__call__(
        extractor,
        [0.0, 0.0],
        return_token_timestamps=True,
    )

    assert torch.equal(processed["num_frames"], torch.tensor([2, 1], dtype=torch.long))


def test_whisper_proxy_exports_mobilint_feature_extractor() -> None:
    """Expose the Mobilint feature extractor name used by the Hub auto_map."""
    assert proxy_whisper.MobilintWhisperFeatureExtractor is MobilintWhisperFeatureExtractor
    assert "MobilintWhisperFeatureExtractor" in proxy_whisper.__all__
    assert "WhisperFeatureExtractor" not in proxy_whisper.__all__
    assert not hasattr(proxy_whisper, "WhisperFeatureExtractor")
