"""Unit tests for Mobilint Qwen2 EAGLE-3 generation output handling."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers.generation.stopping_criteria import StoppingCriteria

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    CachedRotaryEmbedding,
    MobilintQwen2Eagle3ForCausalLM,
)
from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as tree_decoding_module
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache
from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintEagle3GenerationMixin, llm_eagle3_forward
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import evaluate_posterior, update_inference_inputs


def _attach_minimal_eagle3_modules(model: MobilintQwen2Eagle3ForCausalLM) -> None:
    """Attach minimal base/draft/fc modules required by EAGLE-3 helpers."""
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
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache

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
            False,
        ),
    )

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), return_dict_in_generate=True)

    assert torch.equal(output.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))


def test_qwen2_eagle3_acceptance_stats_getter_defaults_and_updates(monkeypatch) -> None:
    """Expose acceptance stats through read-only getter with safe defaults."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    assert model.last_eagle3_acceptance_stats == {
        "steps": 0,
        "accepted_tokens_sum": 0,
        "accepted_tokens_avg": 0.0,
        "acceptance_ratio": 0.0,
    }

    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
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

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), return_dict_in_generate=True)
    stats = model.last_eagle3_acceptance_stats
    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 1
    assert stats["accepted_tokens_avg"] == 1.0
    assert stats["acceptance_ratio"] == 1.0
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
    _attach_minimal_eagle3_modules(model)
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(
        get_seq_length=lambda: 0,
        set_seq_length=lambda _value: None,
    )
    cache.draft_layer = SimpleNamespace(
        get_seq_length=lambda: 0,
        set_seq_length=lambda _value: None,
    )
    cache.get_base_seq_length = lambda: 0
    cache.get_draft_seq_length = lambda: 0
    cache.clear_tree_state = lambda: None
    cache.sync_draft_seq_length_to_base = lambda: None
    cache.reset = lambda: None

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
            False,
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
    _attach_minimal_eagle3_modules(model)
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(
        get_seq_length=lambda: 0,
        set_seq_length=lambda _value: None,
    )
    cache.draft_layer = SimpleNamespace(
        get_seq_length=lambda: 0,
        set_seq_length=lambda _value: None,
    )
    cache.get_base_seq_length = lambda: 0
    cache.get_draft_seq_length = lambda: 0
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

    monkeypatch.setattr(decoding_module, "initialize_tree", _initialize_tree)
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
            False,
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
    _attach_minimal_eagle3_modules(model)
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

    monkeypatch.setattr(decoding_module, "initialize_tree", _initialize_tree)
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
            False,
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
    _attach_minimal_eagle3_modules(model)
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(get_seq_length=lambda: 0, set_seq_length=lambda _value: None)
    cache.draft_layer = SimpleNamespace(get_seq_length=lambda: 0, set_seq_length=lambda _value: None)
    cache.get_base_seq_length = lambda: 0
    cache.set_draft_seq_length = lambda _value: None
    cache.clear_tree_state = lambda: None
    cache.reset = lambda: None

    streamer = RecordingStreamer()

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
            False,
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


def test_qwen2_eagle3_generate_signature_exposes_benchmark_kwargs() -> None:
    """Expose HF and Mobilint generation kwargs used by TPS benchmarks."""
    signature = inspect.signature(MobilintQwen2Eagle3ForCausalLM.generate)

    assert "stopping_criteria" in signature.parameters
    assert "count_npu_time" in signature.parameters
    assert "prefill_chunk_size" in signature.parameters


def test_qwen2_eagle3_generate_calls_stopping_criteria(monkeypatch) -> None:
    """Run custom stopping criteria after EAGLE-3 appends generated tokens."""

    class RecordingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []
            self.score_shapes: list[tuple[int, ...]] = []

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: object) -> bool:
            del kwargs
            self.calls.append(input_ids.clone())
            assert isinstance(scores, torch.Tensor)
            self.score_shapes.append(tuple(scores.shape))
            return True

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
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
    criteria = RecordingCriteria()

    monkeypatch.setattr(
        decoding_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (torch.tensor([[3]]), torch.tensor([0]), None, torch.tensor([[0]]), None),
    )
    monkeypatch.setattr(
        decoding_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (torch.zeros((1, 1, 16)), torch.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(
        decoding_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (torch.tensor(0), torch.tensor(0), torch.zeros(16), None),
    )
    monkeypatch.setattr(
        decoding_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0]),
            None,
            torch.tensor([[0]]),
            1,
            False,
        ),
    )

    output = model.generate(torch.tensor([[1, 2]], dtype=torch.long), stopping_criteria=[criteria])

    assert torch.equal(output, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert len(criteria.calls) == 1
    assert torch.equal(criteria.calls[0], torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert criteria.score_shapes == [(1, 16)]


def test_qwen2_eagle3_generation_config_updates_max_draft_tokens() -> None:
    """`num_assistant_tokens`가 draft 모델의 `max_draft_tokens`에 반영되어야 한다."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.generation_config = SimpleNamespace(
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_assistant_tokens=9,
    )
    draft_model = SimpleNamespace(max_draft_tokens=0)
    object.__setattr__(model, "_modules", {"model": SimpleNamespace(_modules={"draft_model": draft_model})})

    model._resolve_eagle3_generation_config(
        None,
        prompt_length=1,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    assert draft_model.max_draft_tokens == 8


def test_qwen2_eagle3_generate_resolves_greedy_and_sampling_processors(monkeypatch) -> None:
    """Use greedy by default and sampling only when explicitly enabled."""
    temperatures: list[float] = []

    def _prepare_logits_processor(*, temperature: float, top_p: float, top_k: int):
        del top_p, top_k
        temperatures.append(temperature)
        return None

    def _run_once(do_sample: bool | None, config_do_sample: bool) -> None:
        model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
        model.config = SimpleNamespace(vocab_size=16)
        model.generation_config = SimpleNamespace(
            max_new_tokens=1,
            do_sample=config_do_sample,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
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
        model.generate(torch.tensor([[1, 2]], dtype=torch.long), do_sample=do_sample)

    monkeypatch.setattr(decoding_module, "prepare_logits_processor", _prepare_logits_processor)
    monkeypatch.setattr(
        decoding_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (torch.tensor([[3]]), torch.tensor([0]), None, torch.tensor([[0]]), None),
    )
    monkeypatch.setattr(
        decoding_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (torch.zeros((1, 1, 16)), torch.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(
        decoding_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (torch.tensor(0), torch.tensor(0), torch.zeros(16), None),
    )
    monkeypatch.setattr(
        decoding_module,
        "update_inference_inputs",
        lambda *_args, **_kwargs: (
            torch.tensor([[1, 2, 4]], dtype=torch.long),
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0]),
            None,
            torch.tensor([[0]]),
            1,
            True,
        ),
    )

    _run_once(None, False)
    _run_once(True, False)

    assert temperatures == [0.0, 1.0]


def test_qwen2_eagle3_update_inference_inputs_caps_remaining_tokens() -> None:
    """Truncate an accepted EAGLE-3 block to the remaining generation budget."""
    cache = SimpleNamespace(accept_tokens=None)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    candidates = torch.tensor([[4, 5, 6]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hidden_state_new = torch.zeros((1, 3, 1), dtype=torch.float32)

    output, *_rest, new_token_count, should_stop = update_inference_inputs(
        input_ids,
        candidates,
        torch.tensor(0),
        torch.tensor(2),
        retrieve_indices,
        None,
        0,
        SimpleNamespace(),
        cache,
        hidden_state_new,
        torch.zeros(8),
        None,
        remaining_tokens=2,
    )

    assert torch.equal(output, torch.tensor([[1, 2, 4, 5]], dtype=torch.long))
    assert torch.equal(cache.accept_tokens, torch.tensor([[4, 5]], dtype=torch.long))
    assert new_token_count == 2
    assert should_stop is True


def test_qwen2_eagle3_update_inference_inputs_stops_at_mid_block_eos() -> None:
    """Stop at the first EOS token inside a multi-token accepted block."""
    cache = SimpleNamespace(accept_tokens=None)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    candidates = torch.tensor([[4, 5, 6]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hidden_state_new = torch.zeros((1, 3, 1), dtype=torch.float32)

    output, *_rest, new_token_count, should_stop = update_inference_inputs(
        input_ids,
        candidates,
        torch.tensor(0),
        torch.tensor(2),
        retrieve_indices,
        None,
        0,
        SimpleNamespace(),
        cache,
        hidden_state_new,
        torch.zeros(8),
        None,
        remaining_tokens=8,
        eos_token_id=5,
    )

    assert torch.equal(output, torch.tensor([[1, 2, 4, 5]], dtype=torch.long))
    assert new_token_count == 2
    assert should_stop is True


def test_qwen2_eagle3_evaluate_posterior_handles_greedy_full_accept() -> None:
    """Avoid one-past-end logits indexing when every draft token is accepted."""
    logits = torch.zeros((2, 8), dtype=torch.float32)
    logits[0, 4] = 10.0
    logits[1, 5] = 10.0
    candidates = torch.tensor([[3, 4, 5]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, -1]], dtype=torch.long)

    best_candidate, accepted_draft_count, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        None,
        retrieve_indices,
    )

    assert best_candidate.item() == 0
    assert accepted_draft_count.item() == 2
    assert sample_p.shape == (8,)
    assert sampled_indices is None


def test_qwen2_eagle3_evaluate_posterior_sampling_accepts_with_torch_rng(monkeypatch) -> None:
    """Use torch RNG for sampling-path posterior acceptance."""
    logits = torch.zeros((2, 8), dtype=torch.float32)
    candidates = torch.tensor([[3, 4, 5]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, -1]], dtype=torch.long)

    monkeypatch.setattr(
        tree_decoding_module,
        "softmax_topk_cpu_torch",
        lambda *_args, **_kwargs: (torch.tensor([1.0]), torch.tensor([4])),
    )
    monkeypatch.setattr(tree_decoding_module.torch, "rand", lambda *_args, **_kwargs: torch.tensor(0.0))

    best_candidate, accepted_draft_count, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        [object()],
        retrieve_indices,
    )

    assert best_candidate.item() == 0
    assert accepted_draft_count.item() == 2
    assert torch.equal(sample_p, torch.tensor([1.0]))
    assert torch.equal(sampled_indices, torch.tensor([4]))


def test_qwen2_eagle3_generate_multi_turn_reuses_cache_safely(monkeypatch) -> None:
    """Reuse a single cache across turns while clearing tree state and resyncing draft length."""

    class FakeEagle3Cache(MobilintEagle3Cache):
        """Lightweight cache stub that satisfies the type check."""

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = SimpleNamespace(vocab_size=16)
    model.generation_config = SimpleNamespace(
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        eos_token_id=None,
        num_assistant_tokens=2,
    )
    _attach_minimal_eagle3_modules(model)
    cache = object.__new__(FakeEagle3Cache)
    cache.base_layer = SimpleNamespace(get_seq_length=lambda: getattr(cache, "_base_seq", 4))
    cache.draft_layer = SimpleNamespace(
        get_seq_length=lambda: getattr(cache, "_draft_seq", 9),
        set_seq_length=lambda value: setattr(cache, "_draft_seq", int(value)),
    )
    cache.get_base_seq_length = lambda: cache.base_layer.get_seq_length()
    cache.get_draft_seq_length = lambda: cache.draft_layer.get_seq_length()
    cache.sync_draft_seq_length_to_base = lambda: cache.draft_layer.set_seq_length(cache.get_base_seq_length())
    cache.clear_tree_state = lambda: (
        setattr(cache, "accept_tokens", None),
        setattr(cache, "tree_mask", None),
        setattr(cache, "retrieve_indices", None),
        setattr(cache, "tree_position_ids", None),
        setattr(cache, "pending_draft_tokens", None),
    )
    cache.reset = lambda: None
    cache._base_seq = 4
    cache._draft_seq = 9
    cache.accept_tokens = torch.ones(1, 1, dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 1, 1)
    cache.retrieve_indices = torch.ones(1, 1, dtype=torch.long)
    cache.tree_position_ids = torch.ones(1, dtype=torch.long)
    cache.pending_draft_tokens = torch.ones(1, 1, dtype=torch.long)

    init_calls = {"count": 0}

    def _initialize_tree(*_args, **_kwargs):
        init_calls["count"] += 1
        assert cache.get_draft_seq_length() == cache.get_base_seq_length()
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

    monkeypatch.setattr(decoding_module, "initialize_tree", _initialize_tree)
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

    first = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        return_dict_in_generate=True,
    )
    assert torch.equal(first.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))

    cache._draft_seq = 99
    cache.accept_tokens = torch.ones(1, 1, dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 1, 1)
    second = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        past_key_values=cache,
        return_dict_in_generate=True,
    )

    assert torch.equal(second.sequences, torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert init_calls["count"] == 2


def test_cached_rotary_embedding_expands_position_table() -> None:
    """Expand cached RoPE tables when runtime position ids exceed the initial max length."""
    rope = CachedRotaryEmbedding(dim=8, max_position_embeddings=2)
    x = torch.zeros((1, 1, 8), dtype=torch.float32)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    table = rope(x, position_ids)

    assert rope.max_seq_len == 4
    assert table.shape[2] == 4
    assert table.shape[-1] % 128 == 0


def test_qwen2_eagle3_npu_timing_aggregation_and_reset() -> None:
    """Aggregate and reset NPU timing counters across base/draft/fc backends."""

    class _TimingChild:
        def __init__(self, timing: dict[str, float | int]) -> None:
            self._timing = timing
            self.reset_called = False

        def get_npu_timing(self) -> dict[str, float | int]:
            return self._timing

        def reset_npu_timing(self) -> None:
            self.reset_called = True

    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    base = _TimingChild({"prefill_time": 1.0, "decode_time": 2.0, "prefill_calls": 3, "decode_calls": 4})
    draft = _TimingChild({"prefill_time": 0.5, "decode_time": 1.0, "prefill_calls": 1, "decode_calls": 2})
    fc = _TimingChild({"prefill_time": 0.25, "decode_time": 0.75, "prefill_calls": 2, "decode_calls": 1})
    eagle3_model = SimpleNamespace(
        _modules={"base_model": base, "draft_model": draft, "fc_projector": fc},
        fc_projector=fc,
    )
    object.__setattr__(model, "_modules", {"model": eagle3_model})

    timing = model.get_npu_timing()
    model.reset_npu_timing()

    assert timing == {
        "prefill_time": 1.75,
        "decode_time": 3.75,
        "prefill_calls": 6,
        "decode_calls": 7,
    }
    assert base.reset_called is True
    assert draft.reset_called is True
    assert fc.reset_called is True


def test_qwen2_eagle3_npu_timing_requires_all_components() -> None:
    """Fail fast when any EAGLE-3 child backend is missing."""
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    object.__setattr__(model, "_modules", {"model": SimpleNamespace(_modules={"base_model": object(), "draft_model": object()})})

    import pytest

    with pytest.raises(ValueError, match="requires all child backends"):
        model.get_npu_timing()


def test_eagle3_validate_generate_request_ignores_unknown_kwargs_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Ignore unknown generate kwargs and emit a warning instead of raising."""

    class _DummyEagle3(MobilintEagle3GenerationMixin):
        pass

    model = _DummyEagle3()

    with caplog.at_level("WARNING"):
        model._validate_eagle3_generate_request(
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
            num_beams=1,
            assistant_model=None,
            use_cache=True,
            synced_gpus=None,
            logits_processor_arg=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            kwargs={"foo": 1, "bar": 2},
        )

    assert "Unsupported generate kwargs are ignored for EAGLE-3 models: bar, foo" in caplog.text


def test_llm_eagle3_forward_accepts_dict_hidden_states() -> None:
    """Return tuple hidden_states when base output is dict-like."""

    class _DummyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(vocab_size=8)

        def _get_cache(self, *_args, **_kwargs):
            return object.__new__(MobilintEagle3Cache)

        def eagle3_base_model(self, **kwargs):
            del kwargs
            outputs = {"hidden_states": [torch.zeros((1, 1, 4)), torch.ones((1, 1, 4))]}
            logits = torch.zeros((1, 1, 8), dtype=torch.float32)
            return outputs, logits

        def loss_function(self, **kwargs):
            del kwargs
            return torch.tensor(0.0)

    model = _DummyModel()
    cache = object.__new__(MobilintEagle3Cache)

    output = llm_eagle3_forward(model, input_ids=torch.tensor([[1]], dtype=torch.long), past_key_values=cache)

    assert output.hidden_states is not None
    assert isinstance(output.hidden_states, tuple)
    assert len(output.hidden_states) == 2


def test_llm_eagle3_forward_accepts_object_hidden_states_and_missing_hidden_states() -> None:
    """Handle object outputs with/without hidden_states attribute."""

    class _DummyOutput:
        def __init__(self, hidden_states=None) -> None:
            self.hidden_states = hidden_states

    class _DummyModel:
        def __init__(self, with_hidden_states: bool) -> None:
            self.with_hidden_states = with_hidden_states
            self.config = SimpleNamespace(vocab_size=8)

        def _get_cache(self, *_args, **_kwargs):
            return object.__new__(MobilintEagle3Cache)

        def eagle3_base_model(self, **kwargs):
            del kwargs
            hidden_states = [torch.zeros((1, 1, 4))] if self.with_hidden_states else None
            outputs = _DummyOutput(hidden_states=hidden_states)
            logits = torch.zeros((1, 1, 8), dtype=torch.float32)
            return outputs, logits

        def loss_function(self, **kwargs):
            del kwargs
            return torch.tensor(0.0)

    cache = object.__new__(MobilintEagle3Cache)
    with_hidden = llm_eagle3_forward(
        _DummyModel(with_hidden_states=True),
        input_ids=torch.tensor([[1]], dtype=torch.long),
        past_key_values=cache,
    )
    without_hidden = llm_eagle3_forward(
        _DummyModel(with_hidden_states=False),
        input_ids=torch.tensor([[1]], dtype=torch.long),
        past_key_values=cache,
    )

    assert isinstance(with_hidden.hidden_states, tuple)
    assert len(with_hidden.hidden_states) == 1
    assert without_hidden.hidden_states is None


def test_qwen2_eagle3_generate_ignored_args_emit_stable_warning_messages(monkeypatch, caplog) -> None:
    """Warn with explicit ignored-argument policy messages for unsupported soft options."""
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
    _attach_minimal_eagle3_modules(model)
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache

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

    with caplog.at_level("WARNING"):
        model.generate(
            torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            min_new_tokens=1,
            pad_token_id=0,
            prefill_chunk_size=16,
            cache_position=torch.tensor([0], dtype=torch.long),
        )

    joined = "\n".join(caplog.messages)
    assert "attention_mask is not supported and will be ignored." in joined
    assert "min_new_tokens is not supported and will be ignored." in joined
    assert "pad_token_id is not supported and will be ignored." in joined
    assert "prefill_chunk_size is not supported by EAGLE-3 generate and will be ignored." in joined
    assert "cache_position is not supported and will be ignored." in joined


def test_mobilint_eagle3_cache_copy_drops_transient_tree_state() -> None:
    """Copy should preserve KV state while resetting speculative tree metadata."""

    class _Layer:
        def __init__(self, seq: int) -> None:
            self.seq = seq

        def copy(self):
            return _Layer(self.seq)

        def get_seq_length(self) -> int:
            return self.seq

        def set_seq_length(self, sequence_length: int) -> None:
            self.seq = int(sequence_length)

    cache = object.__new__(MobilintEagle3Cache)
    cache.base_mxq_model = object()
    cache.draft_mxq_model = object()
    cache.base_layer = _Layer(7)
    cache.draft_layer = _Layer(5)
    cache.layers = [cache.base_layer]
    cache.accept_tokens = torch.ones(1, 2, dtype=torch.long)
    cache.tree_mask = torch.ones(1, 1, 2, 2)
    cache.retrieve_indices = torch.ones(1, 2, dtype=torch.long)
    cache.tree_position_ids = torch.ones(2, dtype=torch.long)
    cache.pending_draft_tokens = torch.ones(1, 2, dtype=torch.long)

    copied = cache.copy()

    assert copied.get_base_seq_length() == 7
    assert copied.get_draft_seq_length() == 5
    assert copied.accept_tokens is None
    assert copied.tree_mask is None
    assert copied.retrieve_indices is None
    assert copied.tree_position_ids is None
    assert copied.pending_draft_tokens is None
