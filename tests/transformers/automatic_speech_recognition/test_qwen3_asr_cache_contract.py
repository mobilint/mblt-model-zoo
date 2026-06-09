"""Regression tests for Qwen3-ASR cache creation semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

pytest.importorskip(
    "qwen_asr",
    reason="Qwen3-ASR cache contract tests require the optional qwen-asr package.",
)

from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (  # noqa: E402
    MobilintQwen3ASRTextModel,
    MobilintQwen3ASRThinkerForConditionalGeneration,
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


class _DummyQwen3ASRThinkerTextModel(torch.nn.Module):
    """Minimal nested text model for thinker output contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs: object) -> BaseModelOutputWithPast:
        """Return deterministic hidden states while recording forwarded kwargs."""
        self.forward_kwargs = kwargs
        inputs_embeds = kwargs["inputs_embeds"]
        assert isinstance(inputs_embeds, torch.Tensor)
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds, past_key_values=None)


class _DummyQwen3ASRThinker(MobilintQwen3ASRThinkerForConditionalGeneration):
    """Minimal Qwen3-ASR thinker that avoids NPU initialization."""

    def __init__(self, return_dict: bool) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(return_dict=return_dict)
        self.model = _DummyQwen3ASRThinkerTextModel()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Identity()
        self.rope_deltas = None

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return dummy token embeddings."""
        return self.embed_tokens


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


def test_qwen3_asr_thinker_forward_supports_return_dict_false() -> None:
    """Preserve upstream tuple output contract when ``return_dict=False`` is explicit."""
    model = _DummyQwen3ASRThinker(return_dict=True)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    outputs = model(input_ids=input_ids, return_dict=False)

    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], model.get_input_embeddings()(input_ids))
    assert "return_dict" not in model.model.forward_kwargs


def test_qwen3_asr_thinker_forward_uses_config_return_dict_default() -> None:
    """Use the thinker config default when ``return_dict`` is omitted."""
    model = _DummyQwen3ASRThinker(return_dict=False)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    outputs = model(input_ids=input_ids)

    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], model.get_input_embeddings()(input_ids))
