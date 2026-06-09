"""Regression tests for Qwen3-ASR cache creation semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "qwen_asr",
    reason="Qwen3-ASR cache contract tests require the optional qwen-asr package.",
)

from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (  # noqa: E402
    MobilintQwen3ASRTextModel,
)


class _DummyCache:
    """Tiny cache stub exposing the sequence length API used by ``forward``."""

    def get_seq_length(self) -> int:
        """Return an empty cache length."""
        return 0


class _DummyQwen3ASRTextModel(MobilintQwen3ASRTextModel):
    """Minimal Qwen3-ASR text model that avoids NPU initialization."""

    def __init__(self, use_cache: bool) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(use_cache=use_cache, vocab_size=8, hidden_size=4, pad_token_id=0)
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
        )
        self.cache_created = False
        self.forward_past_key_values = None

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args: object) -> object:
        """Record cache creation without allocating a real Mobilint cache."""
        del cache_implementation, batch_size, max_cache_len, args
        self.cache_created = True
        return _DummyCache()

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: object | None,
        cache_position: torch.Tensor,
        prefill_chunk_size: int | None = None,
        count_npu_time: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return deterministic logits while exposing forward-time cache state."""
        del cache_position, prefill_chunk_size, count_npu_time, attention_mask
        self.forward_past_key_values = past_key_values
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.config.vocab_size,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )


@pytest.mark.parametrize("config_use_cache", [False, True])
def test_qwen3_asr_forward_does_not_force_cache_when_use_cache_is_false(config_use_cache: bool) -> None:
    """Respect explicit ``use_cache=False`` even for beam-expanded batches."""
    model = _DummyQwen3ASRTextModel(use_cache=config_use_cache)

    outputs = model(input_ids=torch.tensor([[1, 2], [1, 3]], dtype=torch.long), use_cache=False)

    assert model.cache_created is False
    assert model.forward_past_key_values is None
    assert outputs.past_key_values is None


def test_qwen3_asr_forward_creates_cache_for_multi_row_default_cache() -> None:
    """Keep the existing multi-row cache path when caching is not explicitly disabled."""
    model = _DummyQwen3ASRTextModel(use_cache=True)

    outputs = model(input_ids=torch.tensor([[1, 2], [1, 3]], dtype=torch.long))

    assert model.cache_created is True
    assert model.forward_past_key_values is outputs.past_key_values
    assert outputs.past_key_values is not None