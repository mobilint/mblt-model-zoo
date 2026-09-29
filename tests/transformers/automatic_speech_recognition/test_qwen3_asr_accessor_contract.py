"""Accessor contract for the outer Qwen3-ASR composite.

``MobilintQwen3ASRForConditionalGeneration`` keeps its language model and audio
tower on ``self.thinker`` rather than ``self.model``, so transformers'
``EmbeddingAccessMixin`` cannot resolve the embedding table by itself and the
upstream composite exposes no ``get_audio_features``. Callers that treat the
composite like the other Mobilint multimodal wrappers (for example vllm-mblt's
worker, which calls both accessors on the loaded model) rely on these
delegations. The tests build the composite without an NPU or a download.
"""

from __future__ import annotations

import pytest
import torch

try:
    import qwen_asr  # noqa: F401
except ImportError:
    pytest.skip(
        "Qwen3-ASR accessor contract tests require the optional qwen-asr package.",
        allow_module_level=True,
    )
except (TypeError, AttributeError) as exc:
    # Same guard as test_qwen3_asr_cache_contract.py: qwen-asr pins one
    # transformers release and fails at class-body evaluation on others.
    pytest.skip(
        f"Qwen3-ASR accessor contract tests skipped: installed qwen-asr is "
        f"incompatible with the installed transformers ({exc}).",
        allow_module_level=True,
    )

from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (  # noqa: E402
    MobilintQwen3ASRForConditionalGeneration,
)


class _RecordingThinker(torch.nn.Module):
    """Thinker stub exposing the two accessors the composite delegates to."""

    def __init__(self) -> None:
        super().__init__()
        # Not named embed_tokens: the real thinker keeps its table one level down, so an override that
        # reached for thinker.embed_tokens directly would pass here but fail on the real model.
        self._table = torch.nn.Embedding(8, 4)
        self.audio_calls: list[tuple[object, object, object]] = []

    def get_input_embeddings(self) -> torch.nn.Module:
        return self._table

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.audio_calls.append((input_features, feature_attention_mask, audio_feature_lengths))
        return torch.ones(2, 4)


def _composite() -> MobilintQwen3ASRForConditionalGeneration:
    """Build the composite around a stub thinker without NPU initialization."""
    model = object.__new__(MobilintQwen3ASRForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.thinker = _RecordingThinker()
    return model


def test_composite_has_no_model_attribute_for_the_mixin_to_find() -> None:
    """Pin why the override exists: the mixin's ``base_model_prefix`` lookup misses."""
    from transformers.modeling_utils import EmbeddingAccessMixin

    assert MobilintQwen3ASRForConditionalGeneration.base_model_prefix == "model"
    with pytest.raises(NotImplementedError):
        EmbeddingAccessMixin.get_input_embeddings(_composite())


def test_composite_get_input_embeddings_resolves_through_the_thinker() -> None:
    model = _composite()

    assert model.get_input_embeddings() is model.thinker._table


def test_composite_get_audio_features_forwards_to_the_thinker() -> None:
    model = _composite()
    input_features = torch.zeros(1, 128, 6)
    feature_attention_mask = torch.ones(1, 6, dtype=torch.long)

    audio_features = model.get_audio_features(
        input_features=input_features, feature_attention_mask=feature_attention_mask
    )

    ((seen_features, seen_mask, seen_lengths),) = model.thinker.audio_calls
    assert seen_features is input_features
    assert seen_mask is feature_attention_mask
    assert seen_lengths is None
    assert audio_features.shape == (2, 4)


def test_composite_get_audio_features_passes_audio_feature_lengths_through() -> None:
    model = _composite()
    lengths = torch.tensor([6])

    model.get_audio_features(torch.zeros(1, 128, 6), audio_feature_lengths=lengths)

    ((_, seen_mask, seen_lengths),) = model.thinker.audio_calls
    assert seen_mask is None
    assert seen_lengths is lengths
