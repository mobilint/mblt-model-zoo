"""Unit tests for EAGLE-3 max_new_tokens resolution behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import MobilintQwen2Eagle3ForCausalLM
from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as decoding_module


def _attach_minimal_eagle3_modules(model: MobilintQwen2Eagle3ForCausalLM) -> None:
    """Attach minimal base/draft/fc modules required by EAGLE-3 helpers."""
    input_embeddings = torch.nn.Embedding(16, 1)
    base = SimpleNamespace(get_input_embeddings=lambda: input_embeddings)
    draft = SimpleNamespace(max_draft_tokens=None)
    eagle3_model = SimpleNamespace(
        _modules={
            "base_model": base,
            "draft_model": draft,
            "fc_projector": SimpleNamespace(),
        },
        draft_model=draft,
        fc_projector=SimpleNamespace(),
    )
    object.__setattr__(model, "_modules", {"model": eagle3_model})


def _patch_minimal_generate_dependencies(monkeypatch) -> None:
    """Patch EAGLE-3 decoding functions with deterministic single-step stubs."""
    monkeypatch.setattr(
        decoding_module,
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
        decoding_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 16), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        decoding_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (
            torch.tensor([4], dtype=torch.long),
            1,
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        decoding_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            None,
            torch.tensor([[0]], dtype=torch.long),
            1,
            True,
        ),
    )


def test_qwen2_eagle3_generate_resolves_max_new_tokens_from_max_length(monkeypatch) -> None:
    """Resolve token budget from max_length - prompt_length when max_new_tokens is unset."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16, max_position_embeddings=4096)
    model.generation_config = SimpleNamespace(
        max_new_tokens=None,
        max_length=4,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache
    _patch_minimal_generate_dependencies(monkeypatch)

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), return_dict_in_generate=True)
    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))


def test_qwen2_eagle3_generate_resolves_max_new_tokens_from_model_max_position(monkeypatch) -> None:
    """Fallback to config.max_position_embeddings when max_length and max_new_tokens are unset."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16, max_position_embeddings=3)
    model.generation_config = SimpleNamespace(
        max_new_tokens=None,
        max_length=None,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache
    _patch_minimal_generate_dependencies(monkeypatch)

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), return_dict_in_generate=True)
    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))


def test_qwen2_eagle3_generate_raises_when_resolved_max_new_tokens_non_positive(monkeypatch) -> None:
    """Raise clear error when resolved max_new_tokens becomes zero or negative."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16, max_position_embeddings=2)
    model.generation_config = SimpleNamespace(
        max_new_tokens=None,
        max_length=None,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache
    _patch_minimal_generate_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="Resolved max_new_tokens must be > 0"):
        model.generate(torch.tensor([[1, 2]], dtype=torch.long))
