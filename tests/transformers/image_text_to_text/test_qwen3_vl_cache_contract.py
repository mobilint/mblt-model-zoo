"""Regression tests for Qwen3-VL cache creation semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLTextModel  # noqa: E402


class _DummyCache:
    """Tiny cache stub exposing the sequence length API used by ``forward``."""

    def get_seq_length(self) -> int:
        """Return an empty cache length."""
        return 0


class _DummyQwen3VLTextModel(MobilintQwen3VLTextModel):
    """Minimal Qwen3-VL text model that avoids NPU initialization."""

    def __init__(self, use_cache: bool) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(use_cache=use_cache, vocab_size=8, hidden_size=4, pad_token_id=0)
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
        )
        self.num_deepstack_layers = 0
        self.cache_created = False

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args: object) -> object:
        """Record cache creation without allocating a real Mobilint cache."""
        del cache_implementation, batch_size, max_cache_len, args
        self.cache_created = True
        return _DummyCache()

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        visual_pos_masks: torch.Tensor | None,
        past_key_values: object | None,
        cache_position: torch.Tensor,
        prefill_chunk_size: int | None = None,
        count_npu_time: bool = False,
        logits_to_keep: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        """Return deterministic logits while exposing forward-time cache state."""
        del deepstack_visual_embeds, visual_pos_masks, cache_position, prefill_chunk_size, count_npu_time
        del logits_to_keep
        self.forward_past_key_values = past_key_values
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.config.vocab_size,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )


@pytest.mark.parametrize("config_use_cache", [False, True])
def test_qwen3_vl_forward_does_not_create_cache_when_use_cache_is_false(config_use_cache: bool) -> None:
    """Respect explicit ``use_cache=False`` even when the model config enables caching."""
    model = _DummyQwen3VLTextModel(use_cache=config_use_cache)

    outputs = model(input_ids=torch.tensor([[1, 2]], dtype=torch.long), use_cache=False)

    assert model.cache_created is False
    assert model.forward_past_key_values is None
    assert outputs.past_key_values is None


def test_qwen3_vl_forward_creates_cache_when_use_cache_defaults_to_true() -> None:
    """Create a cache from the config default only when caching is enabled."""
    model = _DummyQwen3VLTextModel(use_cache=True)

    outputs = model(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    assert model.cache_created is True
    assert model.forward_past_key_values is outputs.past_key_values
    assert outputs.past_key_values is not None


def test_qwen3_vl_forward_does_not_create_cache_when_config_disables_cache() -> None:
    """Respect ``config.use_cache=False`` when ``use_cache`` is omitted."""
    model = _DummyQwen3VLTextModel(use_cache=False)

    outputs = model(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    assert model.cache_created is False
    assert model.forward_past_key_values is None
    assert outputs.past_key_values is None
