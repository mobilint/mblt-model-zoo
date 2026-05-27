"""Unit tests for Mobilint Qwen2 EAGLE-3 generation output handling."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3 import modeling_qwen2_eagle3 as eagle3_module
from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3ForCausalLM,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache


def test_qwen2_eagle3_generate_returns_hf_output_when_requested(monkeypatch) -> None:
    """Return a HF generation object when `return_dict_in_generate=True`."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = SimpleNamespace(reset=lambda: None)
    model._get_cache = lambda *_args, **_kwargs: cache

    monkeypatch.setattr(
        eagle3_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
        ),
    )

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), return_dict_in_generate=True)

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert output.past_key_values is cache


def test_qwen2_eagle3_generate_accepts_past_key_values(monkeypatch) -> None:
    """Accept a prefilled EAGLE-3 cache passed through Hugging Face chat serving."""

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = object.__new__(FakeEagle3Cache)

    monkeypatch.setattr(
        eagle3_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
        ),
    )

    output = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        return_dict_in_generate=True,
    )

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert output.past_key_values is cache


def test_qwen2_eagle3_generate_clears_stale_tree_state(monkeypatch) -> None:
    """Drop speculative tree state before starting a new generation call."""

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = object.__new__(FakeEagle3Cache)
    cache.accept_tokens = torch.ones(1, 2, dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 2, 2)
    cache.retrieve_indices = torch.ones(1, 2, dtype=torch.long)
    cache.tree_position_ids = torch.ones(2, dtype=torch.long)
    cache.pending_draft_tokens = torch.ones(1, 2, dtype=torch.long)

    def _initialize_tree(*_args, **_kwargs):
        assert cache.accept_tokens is None
        assert cache.tree_mask is None
        assert cache.retrieve_indices is None
        assert cache.tree_position_ids is None
        assert cache.pending_draft_tokens is None
        return (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            None,
        )

    monkeypatch.setattr(eagle3_module, "initialize_tree", _initialize_tree)
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
        ),
    )

    output = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        return_dict_in_generate=True,
    )

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert output.past_key_values is cache
    assert cache.accept_tokens is None
    assert cache.tree_mask is None
    assert cache.retrieve_indices is None
    assert cache.tree_position_ids is None
    assert cache.pending_draft_tokens is None


def test_qwen2_eagle3_generate_resets_draft_length_to_committed_base(monkeypatch) -> None:
    """Use the committed base length as the draft starting point for the next turn."""

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(
        get_seq_length=lambda: 12,
        set_seq_length=lambda _value: None,
    )
    cache.draft_layer = SimpleNamespace(
        get_seq_length=lambda: 13,
        set_seq_length=lambda value: setattr(cache, "_draft_seq_length", value),
    )
    cache.get_base_seq_length = lambda: 12
    cache.get_draft_seq_length = lambda: getattr(cache, "_draft_seq_length", 13)
    cache.clear_tree_state = lambda: None
    cache.sync_draft_seq_length_to_base = lambda: cache.draft_layer.set_seq_length(cache.get_base_seq_length())
    cache.reset = lambda: None
    cache._draft_seq_length = 13

    def _initialize_tree(*_args, **_kwargs):
        assert cache.get_draft_seq_length() == cache.get_base_seq_length()
        return (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            None,
        )

    monkeypatch.setattr(eagle3_module, "initialize_tree", _initialize_tree)
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
        ),
    )

    output = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        return_dict_in_generate=True,
    )

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert cache.get_draft_seq_length() == cache.get_base_seq_length()


def test_qwen2_eagle3_generate_primes_streamer_with_prompt(monkeypatch) -> None:
    """Prime the streamer with the prompt so the first generated token is not skipped."""

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    class RecordingStreamer:
        def __init__(self) -> None:
            self.values: list[torch.Tensor] = []
            self.ended = False

        def put(self, value: torch.Tensor) -> None:
            self.values.append(value.clone())

        def end(self) -> None:
            self.ended = True

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(get_seq_length=lambda: 0, set_seq_length=lambda _value: None)
    cache.draft_layer = SimpleNamespace(get_seq_length=lambda: 0, set_seq_length=lambda _value: None)
    cache.get_base_seq_length = lambda: 0
    cache.set_draft_seq_length = lambda _value: None
    cache.clear_tree_state = lambda: None
    cache.reset = lambda: None

    streamer = RecordingStreamer()

    monkeypatch.setattr(
        eagle3_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        eagle3_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
        ),
    )

    output = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        streamer=streamer,
        return_dict_in_generate=True,
    )

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert len(streamer.values) == 2
    assert torch.equal(streamer.values[0], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(streamer.values[1], torch.tensor([4], dtype=torch.long))
    assert streamer.ended is True
