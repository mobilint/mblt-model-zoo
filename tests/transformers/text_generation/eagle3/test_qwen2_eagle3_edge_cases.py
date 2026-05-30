"""Edge-case tests for EAGLE-3 decoding and generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3ForCausalLM,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache
from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as tree_decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import evaluate_posterior


def _attach_minimal_eagle3_modules(model: MobilintQwen2Eagle3ForCausalLM) -> None:
    draft = SimpleNamespace(max_draft_tokens=None)
    eagle3_model = SimpleNamespace(
        _modules={
            "base_model": SimpleNamespace(),
            "draft_model": draft,
            "fc_projector": SimpleNamespace(),
        },
        draft_model=draft,
        fc_projector=SimpleNamespace(),
    )
    object.__setattr__(model, "_modules", {"model": eagle3_model})


def _patch_minimal_generate_dependencies(monkeypatch) -> None:
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


def test_evaluate_posterior_sampling_zero_sum_probs_fallback(monkeypatch) -> None:
    """Sampling posterior should avoid NaN when masked probs sum to zero."""
    logits = torch.zeros((2, 8), dtype=torch.float32)
    candidates = torch.tensor([[3, 4, -1], [3, 4, -1]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, -1], [0, 1, -1]], dtype=torch.long)

    monkeypatch.setattr(
        tree_decoding_module,
        "softmax_topk_cpu_torch",
        lambda *_args, **_kwargs: (torch.tensor([1.0], dtype=torch.float32), torch.tensor([4], dtype=torch.long)),
    )
    monkeypatch.setattr(tree_decoding_module.torch, "rand", lambda *_args, **_kwargs: torch.tensor(1.0))

    best_candidate, accepted_draft_count, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        [object()],
        retrieve_indices,
    )

    assert torch.isfinite(sample_p).all()
    assert best_candidate.item() == 0
    assert accepted_draft_count.item() >= 0
    assert sampled_indices is not None


def test_generate_raises_for_empty_prompt_delta_with_reused_cache(monkeypatch) -> None:
    """Reuse cache with no new token delta should raise a clear error."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(get_seq_length=lambda: 2, set_seq_length=lambda _value: None)
    cache.draft_layer = SimpleNamespace(get_seq_length=lambda: 2, set_seq_length=lambda _value: None)
    cache.get_base_seq_length = lambda: 2
    cache.get_draft_seq_length = lambda: 2
    cache.clear_tree_state = lambda: None
    cache.sync_draft_seq_length_to_base = lambda: None
    cache.reset = lambda: None
    model._get_cache = lambda *_args, **_kwargs: cache

    monkeypatch.setattr(
        decoding_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("EAGLE-3 generate received empty prompt delta. When reusing `past_key_values`, provide at least one new input token.")
        ),
    )

    with pytest.raises(ValueError, match="empty prompt delta"):
        model.generate(torch.tensor([[1, 2]], dtype=torch.long), past_key_values=cache)


def test_generate_stops_with_eos_list(monkeypatch) -> None:
    """EOS list should stop generation when one of EOS ids appears."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=3,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        eos_token_id=[5, 7],
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(reset=lambda: None, clear_tree_state=lambda: None, sync_draft_seq_length_to_base=lambda: None)
    model._get_cache = lambda *_args, **_kwargs: cache
    _patch_minimal_generate_dependencies(monkeypatch)

    out = model.generate(torch.tensor([[1, 2]], dtype=torch.long), eos_token_id=[5, 7], return_dict_in_generate=True)
    assert out.sequences.shape[1] >= 3
