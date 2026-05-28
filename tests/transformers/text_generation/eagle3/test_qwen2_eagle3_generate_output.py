"""Unit tests for Mobilint Qwen2 EAGLE-3 generation output handling."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from transformers.generation.stopping_criteria import StoppingCriteria

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3ForCausalLM,
)
from mblt_model_zoo.hf_transformers.utils import eagle3_utils as eagle3_module
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache
from mblt_model_zoo.hf_transformers.utils.eagle3_utils import evaluate_posterior, update_inference_inputs


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
            False,
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

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: object) -> bool:
            del scores, kwargs
            self.calls.append(input_ids.clone())
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
    object.__setattr__(
        model,
        "_modules",
        {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
    )
    cache = SimpleNamespace(reset=lambda: None)
    model._get_cache = lambda *_args, **_kwargs: cache
    criteria = RecordingCriteria()

    monkeypatch.setattr(
        eagle3_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (torch.tensor([[3]]), torch.tensor([0]), None, torch.tensor([[0]]), None),
    )
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (torch.zeros((1, 1, 16)), torch.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (torch.tensor(0), torch.tensor(0), torch.zeros(16), None),
    )
    monkeypatch.setattr(
        eagle3_module,
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
        object.__setattr__(
            model,
            "_modules",
            {"model": SimpleNamespace(_modules={"draft_model": SimpleNamespace(total_tokens=None)})},
        )
        cache = SimpleNamespace(reset=lambda: None)
        model._get_cache = lambda *_args, **_kwargs: cache
        model.generate(torch.tensor([[1, 2]], dtype=torch.long), do_sample=do_sample)

    monkeypatch.setattr(eagle3_module, "prepare_logits_processor", _prepare_logits_processor)
    monkeypatch.setattr(
        eagle3_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (torch.tensor([[3]]), torch.tensor([0]), None, torch.tensor([[0]]), None),
    )
    monkeypatch.setattr(
        eagle3_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (torch.zeros((1, 1, 16)), torch.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(
        eagle3_module,
        "evaluate_posterior",
        lambda *_args, **_kwargs: (torch.tensor(0), torch.tensor(0), torch.zeros(16), None),
    )
    monkeypatch.setattr(
        eagle3_module,
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

    best_candidate, accept_length, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        None,
        retrieve_indices,
    )

    assert best_candidate.item() == 0
    assert accept_length.item() == 2
    assert sample_p.shape == (8,)
    assert sampled_indices is None


def test_qwen2_eagle3_evaluate_posterior_sampling_accepts_with_torch_rng(monkeypatch) -> None:
    """Use torch RNG for sampling-path posterior acceptance."""
    logits = torch.zeros((2, 8), dtype=torch.float32)
    candidates = torch.tensor([[3, 4, 5]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, -1]], dtype=torch.long)

    monkeypatch.setattr(
        eagle3_module,
        "softmax_topk_cpu_torch",
        lambda *_args, **_kwargs: (torch.tensor([1.0]), torch.tensor([4])),
    )
    monkeypatch.setattr(eagle3_module.torch, "rand", lambda *_args, **_kwargs: torch.tensor(0.0))

    best_candidate, accept_length, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        [object()],
        retrieve_indices,
    )

    assert best_candidate.item() == 0
    assert accept_length.item() == 2
    assert torch.equal(sample_p, torch.tensor([1.0]))
    assert torch.equal(sampled_indices, torch.tensor([4]))
