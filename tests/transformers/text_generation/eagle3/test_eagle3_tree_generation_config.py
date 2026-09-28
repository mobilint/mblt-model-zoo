"""Unit tests for resolving EAGLE-3 draft tree shape and NPU chunk size per generate call."""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import MobilintQwen2Eagle3ForCausalLM
from mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils import MobilintEagle3DraftModelMixin

from .test_qwen2_eagle3_generate_max_new_tokens import _patch_minimal_generate_dependencies


def _make_draft(depth: int = 5, top_k: int = 8) -> SimpleNamespace:
    draft = SimpleNamespace(max_draft_tokens=None)
    draft.set_tree_shape = types.MethodType(MobilintEagle3DraftModelMixin.set_tree_shape, draft)
    draft.set_tree_shape(depth=depth, top_k=top_k)
    return draft


def _make_model(
    *, config: SimpleNamespace, generation_config: SimpleNamespace, draft
) -> MobilintQwen2Eagle3ForCausalLM:
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    model.config = config
    model.generation_config = generation_config
    input_embeddings = torch.nn.Embedding(16, 1)
    eagle3_model = SimpleNamespace(
        _modules={
            "base_model": SimpleNamespace(get_input_embeddings=lambda: input_embeddings),
            "draft_model": draft,
            "fc_projector": SimpleNamespace(),
        },
        draft_model=draft,
        fc_projector=SimpleNamespace(),
    )
    object.__setattr__(model, "_modules", {"model": eagle3_model})
    cache = SimpleNamespace(
        reset=lambda: None, clear_tree_state=lambda: None, sync_draft_seq_length_to_base=lambda: None
    )
    model._get_cache = lambda *_args, **_kwargs: cache
    return model


def _generation_config(**tree_fields: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_new_tokens=1,
        max_length=None,
        temperature=None,
        top_p=None,
        top_k=1,
        eos_token_id=None,
        num_assistant_tokens=16,
        **tree_fields,
    )


def _resolve(model: MobilintQwen2Eagle3ForCausalLM, **kwargs) -> None:
    model._resolve_eagle3_generation_config(
        None,
        prompt_length=2,
        max_new_tokens=None,
        max_length=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        top_k=None,
        **kwargs,
    )


def test_generation_config_tree_fields_override_legacy_config_fields() -> None:
    draft = _make_draft()
    model = _make_model(
        config=SimpleNamespace(vocab_size=16, eagle3_tree_depth=6, eagle3_tree_top_k=8),
        generation_config=_generation_config(eagle3_tree_depth=3, eagle3_tree_top_k=4),
        draft=draft,
    )

    _resolve(model)

    assert (draft.depth, draft.top_k) == (3, 4)
    assert draft.tree_mask_init.shape == (1, 1, 4, 4)
    assert draft.position_ids.shape == (4,)
    assert draft.max_draft_tokens == 15


def test_legacy_config_tree_fields_apply_when_generation_config_lacks_them() -> None:
    draft = _make_draft()
    model = _make_model(
        config=SimpleNamespace(vocab_size=16, eagle3_tree_depth=6, eagle3_tree_top_k=3),
        generation_config=_generation_config(),
        draft=draft,
    )

    _resolve(model)

    assert (draft.depth, draft.top_k) == (6, 3)


def test_tree_fields_fall_back_per_field_and_keep_current_when_unset() -> None:
    draft = _make_draft(depth=5, top_k=8)
    model = _make_model(
        config=SimpleNamespace(vocab_size=16),
        generation_config=_generation_config(eagle3_tree_top_k=2),
        draft=draft,
    )

    _resolve(model)

    assert (draft.depth, draft.top_k) == (5, 2)


def test_explicit_generate_kwargs_override_generation_config(monkeypatch) -> None:
    draft = _make_draft()
    model = _make_model(
        config=SimpleNamespace(vocab_size=16, max_position_embeddings=4096),
        generation_config=_generation_config(eagle3_tree_depth=3, eagle3_tree_top_k=4),
        draft=draft,
    )
    _patch_minimal_generate_dependencies(monkeypatch)

    model.generate(torch.tensor([[1, 2]], dtype=torch.long), eagle3_tree_depth=7, eagle3_tree_top_k=5)

    assert (draft.depth, draft.top_k) == (7, 5)
    assert draft.tree_mask_init.shape == (1, 1, 5, 5)


def test_draft_without_set_tree_shape_is_left_untouched() -> None:
    draft = SimpleNamespace(max_draft_tokens=None)
    model = _make_model(
        config=SimpleNamespace(vocab_size=16),
        generation_config=_generation_config(eagle3_tree_depth=3, eagle3_tree_top_k=4),
        draft=draft,
    )

    _resolve(model)

    assert not hasattr(draft, "depth")


@pytest.mark.parametrize(("depth", "top_k"), [(0, 4), (3, 0), (-1, 2)])
def test_set_tree_shape_rejects_non_positive_values(depth: int, top_k: int) -> None:
    draft = _make_draft()
    with pytest.raises(ValueError, match="must be positive"):
        draft.set_tree_shape(depth=depth, top_k=top_k)


def test_generate_stages_npu_prefill_chunk_size_on_base_and_draft_for_the_call(monkeypatch) -> None:
    draft = _make_draft()
    model = _make_model(
        config=SimpleNamespace(vocab_size=16, max_position_embeddings=4096),
        generation_config=_generation_config(),
        draft=draft,
    )
    base = model.eagle3_base_model
    _patch_minimal_generate_dependencies(monkeypatch)
    seen: dict[str, object] = {}
    original_loop = model._run_eagle3_decode_loop

    def _spy_loop(**kwargs):
        seen["base"] = getattr(base, "npu_prefill_chunk_size_override", None)
        seen["draft"] = getattr(draft, "npu_prefill_chunk_size_override", None)
        return original_loop(**kwargs)

    model._run_eagle3_decode_loop = _spy_loop

    model.generate(torch.tensor([[1, 2]], dtype=torch.long), npu_prefill_chunk_size=32)

    assert seen == {"base": 32, "draft": 32}
    assert base.npu_prefill_chunk_size_override is None
    assert draft.npu_prefill_chunk_size_override is None


def test_generate_restores_npu_prefill_chunk_size_override_on_error(monkeypatch) -> None:
    draft = _make_draft()
    model = _make_model(
        config=SimpleNamespace(vocab_size=16, max_position_embeddings=4096),
        generation_config=_generation_config(),
        draft=draft,
    )
    _patch_minimal_generate_dependencies(monkeypatch)

    def _boom(**_kwargs):
        raise RuntimeError("decode failed")

    model._run_eagle3_decode_loop = _boom

    with pytest.raises(RuntimeError, match="decode failed"):
        model.generate(torch.tensor([[1, 2]], dtype=torch.long), npu_prefill_chunk_size=32)

    assert draft.npu_prefill_chunk_size_override is None
    assert model.eagle3_base_model.npu_prefill_chunk_size_override is None


@pytest.mark.parametrize("source", ["kwarg", "generation_config"])
@pytest.mark.parametrize("value", [1, 0])
def test_num_assistant_tokens_below_two_is_rejected(source: str, value: int) -> None:
    """A round needs the root plus at least one draft token; generate must not silently use 2."""
    draft = _make_draft()
    generation_config = _generation_config()
    kwargs = {}
    if source == "kwarg":
        kwargs["num_assistant_tokens"] = value
    else:
        generation_config.num_assistant_tokens = value
    model = _make_model(config=SimpleNamespace(vocab_size=16), generation_config=generation_config, draft=draft)

    with pytest.raises(ValueError, match="num_assistant_tokens must be >= 2"):
        _resolve(model, **kwargs)
    assert draft.max_draft_tokens is None


def test_num_assistant_tokens_two_maps_to_one_draft_token() -> None:
    draft = _make_draft()
    model = _make_model(config=SimpleNamespace(vocab_size=16), generation_config=_generation_config(), draft=draft)

    _resolve(model, num_assistant_tokens=2)

    assert draft.max_draft_tokens == 1
