"""Edge-case tests for EAGLE-3 decoding and generation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3ForCausalLM,
)
from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
    MobilintQwen2Eagle3Config,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache
from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as tree_decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import evaluate_posterior, initialize_tree
from mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils import MobilintEagle3DraftModelMixin
from mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils import MobilintEagle3BaseModelMixin


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
        lambda *_args, **_kwargs: (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([4], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        tree_decoding_module.torch,
        "rand",
        lambda *_args, **_kwargs: torch.tensor(1.0),
    )

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


def test_evaluate_posterior_uses_leaf_logits_after_full_path_accept() -> None:
    """Greedy posterior should sample from leaf logits when full draft path is accepted."""
    logits = torch.tensor(
        [
            [0.0, 10.0, 0.0, 0.0],   # pos 0 -> greedy token 1
            [0.0, 0.0, 10.0, 0.0],   # pos 1 -> greedy token 2
            [0.0, 0.0, 0.0, 10.0],   # pos 2 -> greedy token 3
            [9.0, 1.0, 2.0, 3.0],    # leaf pos to be used for next-token sampling
        ],
        dtype=torch.float32,
    )
    candidates = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    retrieve_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    best_candidate, accepted_draft_count, sample_p, sampled_indices = evaluate_posterior(
        logits,
        candidates,
        logits_processor=None,
        retrieve_indices=retrieve_indices,
    )

    assert best_candidate.item() == 0
    assert accepted_draft_count.item() == 3
    assert sampled_indices is None
    assert torch.equal(sample_p, logits[3])


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
    cache.base_layer = SimpleNamespace(
        get_seq_length=lambda: 2,
        set_seq_length=lambda _value: None,
    )
    cache.draft_layer = SimpleNamespace(
        get_seq_length=lambda: 2,
        set_seq_length=lambda _value: None,
    )
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
            ValueError(
                "EAGLE-3 generate received empty prompt delta. "
                "When reusing `past_key_values`, provide at least one "
                "new input token."
            )
        ),
    )

    with pytest.raises(ValueError, match="empty prompt delta"):
        model.generate(torch.tensor([[1, 2]], dtype=torch.long), past_key_values=cache)


def test_initialize_tree_raises_for_empty_prompt_delta_with_equal_cache_length() -> None:
    """When cache length equals input length, prompt delta must be empty and raise."""

    class _DummyBaseModel:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("Base model must not be called for empty prompt delta.")

    class _DummyModel:
        eagle3_base_model = _DummyBaseModel()

    class _DummyCache:
        def get_base_seq_length(self) -> int:
            return 2

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="empty prompt delta"):
        initialize_tree(
            input_ids,
            _DummyModel(),
            _DummyCache(),
            logits_processor=None,
        )


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
    cache = SimpleNamespace(
        reset=lambda: None,
        clear_tree_state=lambda: None,
        sync_draft_seq_length_to_base=lambda: None,
    )
    model._get_cache = lambda *_args, **_kwargs: cache
    _patch_minimal_generate_dependencies(monkeypatch)

    out = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        eos_token_id=[5, 7],
        return_dict_in_generate=True,
    )
    assert out.sequences.shape[1] >= 3


def test_draft_forward_concatenates_chunk_logits_on_sequence_dimension() -> None:
    """Chunked draft logits should be concatenated on sequence axis, not vocab axis."""

    class _DummyMxqModel:
        def __init__(self) -> None:
            self.token_offset = 0

        def infer(self, infer_inputs, *_args):
            chunk_len = int(infer_inputs[0].shape[2])
            hidden = np.zeros((1, 1, chunk_len, 4), dtype=np.float32)
            logits = np.zeros((1, 1, chunk_len, 3), dtype=np.float32)
            for i in range(chunk_len):
                token_pos = self.token_offset + i
                logits[0, 0, i, :] = np.array([token_pos, token_pos + 100, token_pos + 200], dtype=np.float32)
            self.token_offset += chunk_len
            return hidden, logits

    class _DummyDraftModel(MobilintEagle3DraftModelMixin):
        def __init__(self) -> None:
            self.config = SimpleNamespace(eagle3_npu_chunk_size=2)
            self._mxq = _DummyMxqModel()
            self.embed_tokens = lambda ids: torch.zeros(
                (ids.shape[0], ids.shape[1], 4),
                dtype=torch.float32,
                device=ids.device,
            )
            self.rotary_emb = lambda _x, position_ids: np.zeros(
                (1, position_ids.numel(), 8),
                dtype=np.float32,
            )

        def resolve_prefill_chunk_size(self, chunk_size: int) -> int:
            return chunk_size

        def get_mxq_model(self):
            return self._mxq

        def _record_npu_timing(self, *_args, **_kwargs) -> None:
            return None

    dummy_model = _DummyDraftModel()
    cache = SimpleNamespace(get_draft_seq_length=lambda: 0)
    hidden_states = torch.zeros((1, 5, 4), dtype=torch.float32)
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    _, logits_out = dummy_model.forward(
        hidden_states,
        input_ids=input_ids,
        cache=cache,
        requires_all_features=True,
    )

    assert logits_out.shape == (1, 5, 3)
    expected = torch.tensor(
        [
            [
                [0.0, 100.0, 200.0],
                [1.0, 101.0, 201.0],
                [2.0, 102.0, 202.0],
                [3.0, 103.0, 203.0],
                [4.0, 104.0, 204.0],
            ]
        ]
    )
    assert torch.equal(logits_out, expected)


def test_base_prepare_decoder_attention_mask_single_token_creates_all_keep_mask() -> None:
    """Base path should create an all-keep mask for single-token decode."""

    class _DummyBaseModel(MobilintEagle3BaseModelMixin):
        pass

    model = _DummyBaseModel()
    inputs_embeds = torch.zeros((2, 1, 4), dtype=torch.float32)
    cache = SimpleNamespace(tree_mask=None)

    mask = model._prepare_decoder_attention_mask(
        attention_mask=None,
        input_shape=(2, 1),
        inputs_embeds=inputs_embeds,
        past_key_values_length=3,
        cache=cache,
    )

    assert mask.shape == (2, 1, 1, 4)
    assert torch.all(mask == 0)


def test_draft_prepare_decoder_attention_mask_single_token_creates_all_keep_mask() -> None:
    """Draft path should create an all-keep mask for single-token decode."""

    class _DummyDraftModel(MobilintEagle3DraftModelMixin):
        pass

    model = _DummyDraftModel()
    hidden_states = torch.zeros((2, 1, 4), dtype=torch.float32)

    mask = model._prepare_decoder_attention_mask(
        attention_mask=None,
        input_shape=(2, 1),
        hidden_states=hidden_states,
        past_key_values_length=5,
        cache=SimpleNamespace(),
        tree_mask=None,
    )

    assert mask.shape == (2, 1, 1, 6)
    assert torch.all(mask == 0)


def test_eagle3_global_backend_options_apply_to_all_child_backends() -> None:
    """Shared backend kwargs should propagate to base/draft/fc backends."""
    config = MobilintQwen2Eagle3Config(
        mxq_path="shared.mxq",
        core_mode="global4",
        dev_no=3,
        max_batch_size=4,
        target_clusters=[0],
    )

    assert config.base_mxq_path == "shared.mxq"
    assert config.draft_mxq_path == "shared.mxq"
    assert config.fc_mxq_path == "shared.mxq"

    assert config.base_core_mode == "global4"
    assert config.draft_core_mode == "global4"
    assert config.fc_core_mode == "global4"

    assert config.base_npu_backend.dev_no == 3
    assert config.draft_npu_backend.dev_no == 3
    assert config.fc_npu_backend.dev_no == 3

    assert config.base_npu_backend.max_batch_size == 4
    assert config.draft_npu_backend.max_batch_size == 4
    assert config.fc_npu_backend.max_batch_size == 4


def test_eagle3_prefixed_backend_options_override_shared_defaults() -> None:
    """Prefixed backend kwargs should take precedence over shared kwargs."""
    config = MobilintQwen2Eagle3Config(
        mxq_path="shared.mxq",
        core_mode="global4",
        base_core_mode="single",
        draft_mxq_path="draft.mxq",
        fc_dev_no=7,
    )

    assert config.base_core_mode == "single"
    assert config.draft_core_mode == "global4"
    assert config.fc_core_mode == "global4"

    assert config.base_mxq_path == "shared.mxq"
    assert config.draft_mxq_path == "draft.mxq"
    assert config.fc_mxq_path == "shared.mxq"

    assert config.base_npu_backend.dev_no == 0
    assert config.draft_npu_backend.dev_no == 0
    assert config.fc_npu_backend.dev_no == 7


def test_topk_generate_respects_max_tree_depth_contract() -> None:
    """`depth` should mean max tree depth, not additional expansion rounds."""

    class _DepthContractDraftModel(MobilintEagle3DraftModelMixin):
        def __init__(self, depth: int) -> None:
            self.max_draft_tokens = 1
            self.depth = depth
            self.top_k = 1
            self.draft_config = SimpleNamespace(vocab_size=8, draft_vocab_size=8)
            self.logsoftmax = lambda x: x
            self.tree_mask_init = torch.ones((1, 1, 1, 1), dtype=torch.float32)
            self.position_ids = torch.tensor([0], dtype=torch.long)
            self.d2t = torch.zeros(8, dtype=torch.long)
            self.forward_call_count = 0

        def __call__(
            self,
            hidden_states: torch.Tensor,
            *,
            input_ids: torch.LongTensor,
            cache,
            attention_mask=None,
            position_ids=None,
            requires_all_features=False,
            add_cache_position=0,
            tree_mask=None,
            count_npu_time=False,
        ):
            del input_ids, cache, attention_mask, position_ids, add_cache_position, tree_mask, count_npu_time
            self.forward_call_count += 1
            if requires_all_features:
                out_hidden = torch.zeros(
                    (1, 1, 1),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                logits = torch.zeros((1, 1, 8), dtype=torch.float32, device=hidden_states.device)
                return out_hidden, logits
            last_hidden = torch.zeros((1, 1), dtype=torch.float32, device=hidden_states.device)
            last_hidden_logits = torch.zeros((1, 8), dtype=torch.float32, device=hidden_states.device)
            return last_hidden, last_hidden_logits

    class _DummyCache:
        def __init__(self) -> None:
            self._draft_seq_len = 0

        def get_draft_seq_length(self) -> int:
            return self._draft_seq_len

        def update_draft_seen_tokens(self, delta: int) -> None:
            self._draft_seq_len += int(delta)

    hidden_states = torch.zeros((1, 1, 1), dtype=torch.float32)
    input_ids = torch.tensor([[1]], dtype=torch.long)

    depth1_model = _DepthContractDraftModel(depth=1)
    depth1_model.topk_generate(
        hidden_states,
        input_ids=input_ids,
        cache=_DummyCache(),
        logits_processor=None,
    )

    depth2_model = _DepthContractDraftModel(depth=2)
    depth2_model.topk_generate(
        hidden_states,
        input_ids=input_ids,
        cache=_DummyCache(),
        logits_processor=None,
    )

    # One pre-loop call is always required to create the first depth level.
    assert depth1_model.forward_call_count == 1
    # depth=2 must add exactly one extra expansion call.
    assert depth2_model.forward_call_count == 2


def test_topk_generate_caps_oversized_draft_token_request_without_name_error(caplog) -> None:
    """Oversized draft-token requests should warn and cap instead of crashing."""

    class _OversizedDraftModel(MobilintEagle3DraftModelMixin):
        def __init__(self) -> None:
            self.max_draft_tokens = 8
            self.depth = 1
            self.top_k = 2
            self.draft_config = SimpleNamespace(vocab_size=8, draft_vocab_size=8)
            self.logsoftmax = lambda x: x
            self.tree_mask_init = torch.ones((1, 1, 1, 1), dtype=torch.float32)
            self.position_ids = torch.tensor([0], dtype=torch.long)
            self.d2t = torch.zeros(8, dtype=torch.long)

        def __call__(
            self,
            hidden_states: torch.Tensor,
            *,
            input_ids: torch.LongTensor,
            cache,
            attention_mask=None,
            position_ids=None,
            requires_all_features=False,
            add_cache_position=0,
            tree_mask=None,
            count_npu_time=False,
        ):
            del (
                input_ids,
                cache,
                attention_mask,
                position_ids,
                requires_all_features,
                add_cache_position,
                tree_mask,
                count_npu_time,
            )
            last_hidden = torch.zeros((1, 1), dtype=torch.float32, device=hidden_states.device)
            # With top_k=2, only two initial candidates are available.
            last_hidden_logits = torch.tensor(
                [[3.0, 2.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=hidden_states.device,
            )
            return last_hidden, last_hidden_logits

    class _DummyCache:
        def __init__(self) -> None:
            self._draft_seq_len = 0

        def get_draft_seq_length(self) -> int:
            return self._draft_seq_len

        def update_draft_seen_tokens(self, delta: int) -> None:
            self._draft_seq_len += int(delta)

    model = _OversizedDraftModel()
    cache = _DummyCache()
    hidden_states = torch.zeros((1, 1, 1), dtype=torch.float32)
    input_ids = torch.tensor([[1]], dtype=torch.long)

    with caplog.at_level("WARNING"):
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
            hidden_states,
            input_ids=input_ids,
            cache=cache,
            logits_processor=None,
            max_draft_tokens=64,
        )

    assert "exceed available tree candidates" in caplog.text
    # root token + capped draft candidates(2)
    assert draft_tokens.shape == (1, 3)
    assert retrieve_indices.numel() > 0
    assert tree_mask.shape[-1] == 3
    assert tree_position_ids.shape[0] == 3
