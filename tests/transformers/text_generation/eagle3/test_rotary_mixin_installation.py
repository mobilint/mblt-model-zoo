"""Regression tests for EAGLE-3 base/draft mixin ``__init__`` installation.

These tests guard the refactor that moved ``embed_tokens`` and ``rotary_emb``
construction from every concrete EAGLE-3 family class into the shared behavior
mixins (:class:`MobilintEagle3BaseModelMixin` and
:class:`MobilintEagle3DraftModelMixin`). They verify:

1. Each concrete ``Base*`` class ends up with a
   :class:`ScaledCachedRotaryEmbedding` (the rotary class every shipped
   EAGLE-3 base uses today, because their configs carry ``rope_scaling``).
2. Each concrete ``Draft*`` class ends up with a
   :class:`CachedRotaryEmbedding`.
3. ``embed_tokens`` reports the correct shapes: base uses ``config`` fields;
   draft uses ``draft_config`` fields.
4. The documented ``_build_base_rotary_emb`` / ``_build_draft_rotary_emb``
   override hooks are respected by the concrete family.
5. ``draft_config.rope_theta`` flows through to
   :attr:`CachedRotaryEmbedding.base`.

The tests stub :meth:`MobilintNPUBackend.create` and
:meth:`MobilintNPUBackend.launch` to no-ops so the ``__init__`` chain runs
without a real MXQ artifact or NPU device. Rotary classes themselves are
not mocked — they are instantiated for real to keep the guard honest.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
    MobilintQwen2Eagle3Config,
)
from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3BaseModel,
    MobilintQwen2Eagle3DraftModel,
)
from mblt_model_zoo.hf_transformers.models.qwen3_eagle3.configuration_qwen3_eagle3 import (
    MobilintQwen3Eagle3Config,
)
from mblt_model_zoo.hf_transformers.models.qwen3_eagle3.modeling_qwen3_eagle3 import (
    MobilintQwen3Eagle3BaseModel,
    MobilintQwen3Eagle3DraftModel,
)
from mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils import (
    CachedRotaryEmbedding,
    MobilintEagle3BaseModelMixin,
    MobilintEagle3DraftModelMixin,
    MobilintEagle3FCProjector,
    ScaledCachedRotaryEmbedding,
)
from mblt_model_zoo.utils.npu_backend import MobilintNPUBackend

_QWEN2_DRAFT_CONFIG: dict[str, object] = {
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
}


_QWEN3_DRAFT_CONFIG: dict[str, object] = {
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
}


# Every EAGLE-3 base config shipped in this repo carries a ``rope_scaling``
# entry, so the scaled-rotary branch is what production hits. This fixture
# picks ``rope_type="default"`` because on HEAD the non-default HF init
# functions (``llama3``, ``yarn``, ...) do not accept the flexible kwargs
# ``ScaledCachedRotaryEmbedding`` passes them (a pre-existing issue fixed on
# a separate branch). We only need a truthy ``rope_scaling`` to steer
# ``_build_base_rotary_emb`` into the scaled branch — the exact rope_type
# does not affect the isinstance assertion this suite guards.
_TRUTHY_DEFAULT_ROPE_SCALING: dict[str, object] = {
    "rope_type": "default",
    "factor": 1.0,
}


@pytest.fixture
def stub_npu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace :class:`MobilintNPUBackend` lifecycle methods with no-ops.

    ``create``/``launch`` normally load and dispatch real MXQ artifacts.
    These tests only care about behavior-mixin ``__init__`` side effects, so
    we stub the whole NPU boot sequence to a no-op.
    """
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: None)
    monkeypatch.setattr(MobilintNPUBackend, "launch", lambda self: None)
    return None


def _make_qwen2_config() -> MobilintQwen2Eagle3Config:
    return MobilintQwen2Eagle3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_scaling=dict(_TRUTHY_DEFAULT_ROPE_SCALING),
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        eagle3_tree_depth=2,
        eagle3_tree_top_k=4,
        eagle3_npu_chunk_size=64,
        draft_config=dict(_QWEN2_DRAFT_CONFIG),
        name_or_path="mobilint/test-qwen2-eagle3-fixture",
    )


def _make_qwen3_config() -> MobilintQwen3Eagle3Config:
    return MobilintQwen3Eagle3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_scaling=dict(_TRUTHY_DEFAULT_ROPE_SCALING),
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        eagle3_tree_depth=2,
        eagle3_tree_top_k=4,
        eagle3_npu_chunk_size=64,
        draft_config=dict(_QWEN3_DRAFT_CONFIG),
        name_or_path="mobilint/test-qwen3-eagle3-fixture",
    )


@pytest.mark.parametrize(
    "config_factory, base_cls",
    [
        (_make_qwen2_config, MobilintQwen2Eagle3BaseModel),
        (_make_qwen3_config, MobilintQwen3Eagle3BaseModel),
    ],
    ids=["qwen2_eagle3", "qwen3_eagle3"],
)
def test_base_model_installs_scaled_rotary_embedding(
    stub_npu_backend: None,
    config_factory,
    base_cls,
) -> None:
    """Base classes shipped with ``rope_scaling`` install the scaled RoPE variant."""
    del stub_npu_backend
    config = config_factory()
    base = base_cls(config, _internal_call=True, no_launch=True)
    assert isinstance(base.rotary_emb, ScaledCachedRotaryEmbedding)


@pytest.mark.parametrize(
    "config_factory, draft_cls",
    [
        (_make_qwen2_config, MobilintQwen2Eagle3DraftModel),
        (_make_qwen3_config, MobilintQwen3Eagle3DraftModel),
    ],
    ids=["qwen2_eagle3", "qwen3_eagle3"],
)
def test_draft_model_installs_cached_rotary_embedding(
    stub_npu_backend: None,
    config_factory,
    draft_cls,
) -> None:
    """Draft classes always install the plain :class:`CachedRotaryEmbedding`."""
    del stub_npu_backend
    config = config_factory()
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = draft_cls(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert isinstance(draft.rotary_emb, CachedRotaryEmbedding)


@pytest.mark.parametrize(
    "config_factory, base_cls",
    [
        (_make_qwen2_config, MobilintQwen2Eagle3BaseModel),
        (_make_qwen3_config, MobilintQwen3Eagle3BaseModel),
    ],
    ids=["qwen2_eagle3", "qwen3_eagle3"],
)
def test_base_embed_tokens_shape_matches_config(
    stub_npu_backend: None,
    config_factory,
    base_cls,
) -> None:
    """Base ``embed_tokens`` reports ``config.vocab_size`` / ``config.hidden_size``."""
    del stub_npu_backend
    config = config_factory()
    base = base_cls(config, _internal_call=True, no_launch=True)
    assert base.embed_tokens.num_embeddings == config.vocab_size
    assert base.embed_tokens.embedding_dim == config.hidden_size


@pytest.mark.parametrize(
    "config_factory, draft_cls",
    [
        (_make_qwen2_config, MobilintQwen2Eagle3DraftModel),
        (_make_qwen3_config, MobilintQwen3Eagle3DraftModel),
    ],
    ids=["qwen2_eagle3", "qwen3_eagle3"],
)
def test_draft_embed_tokens_shape_matches_draft_config(
    stub_npu_backend: None,
    config_factory,
    draft_cls,
) -> None:
    """Draft ``embed_tokens`` reports ``draft_config.vocab_size`` / ``draft_config.hidden_size``."""
    del stub_npu_backend
    config = config_factory()
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = draft_cls(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert draft.embed_tokens.num_embeddings == config.draft_config.vocab_size
    assert draft.embed_tokens.embedding_dim == config.draft_config.hidden_size


def test_base_rotary_override_hook_is_respected(stub_npu_backend: None) -> None:
    """A subclass override of ``_build_base_rotary_emb`` wins over the default."""
    del stub_npu_backend
    sentinel = nn.Identity()

    class _OverriddenBase(MobilintQwen2Eagle3BaseModel):
        def _build_base_rotary_emb(self, config):
            del config
            return sentinel

    config = _make_qwen2_config()
    base = _OverriddenBase(config, _internal_call=True, no_launch=True)
    assert base.rotary_emb is sentinel


def test_draft_rotary_override_hook_is_respected(stub_npu_backend: None) -> None:
    """A subclass override of ``_build_draft_rotary_emb`` wins over the default."""
    del stub_npu_backend
    sentinel = nn.Identity()

    class _OverriddenDraft(MobilintQwen2Eagle3DraftModel):
        def _build_draft_rotary_emb(self, draft_config):
            del draft_config
            return sentinel

    config = _make_qwen2_config()
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = _OverriddenDraft(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert draft.rotary_emb is sentinel


def test_draft_rope_theta_flows_through_to_cached_rope_base(stub_npu_backend: None) -> None:
    """``draft_config.rope_theta`` should become ``CachedRotaryEmbedding.base``."""
    del stub_npu_backend
    draft_kwargs = dict(_QWEN2_DRAFT_CONFIG)
    draft_kwargs["rope_theta"] = 50000
    config = MobilintQwen2Eagle3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_scaling=dict(_TRUTHY_DEFAULT_ROPE_SCALING),
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        eagle3_tree_depth=2,
        eagle3_tree_top_k=4,
        eagle3_npu_chunk_size=64,
        draft_config=draft_kwargs,
        name_or_path="mobilint/test-qwen2-eagle3-rope-theta",
    )
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = MobilintQwen2Eagle3DraftModel(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert isinstance(draft.rotary_emb, CachedRotaryEmbedding)
    assert draft.rotary_emb.base == 50000


def test_base_mixin_falls_back_to_cached_rope_when_rope_scaling_absent() -> None:
    """Fallback branch: no ``rope_scaling`` on config → plain ``CachedRotaryEmbedding``.

    Uses a minimal in-memory subclass so we do not depend on the concrete NPU
    boot chain. This documents the ``_build_base_rotary_emb`` default for
    hypothetical future families that ship without ``rope_scaling``.
    """

    class _NoBackendBase(MobilintEagle3BaseModelMixin, nn.Module):
        def __init__(self, config) -> None:
            nn.Module.__init__(self)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            self.rotary_emb = self._build_base_rotary_emb(config)

    class _MinimalConfig:
        vocab_size = 32
        hidden_size = 16
        num_attention_heads = 2
        pad_token_id = 0
        max_position_embeddings = 128
        rope_scaling = None
        rope_theta = 42

    base = _NoBackendBase(_MinimalConfig())
    assert isinstance(base.rotary_emb, CachedRotaryEmbedding)
    assert base.rotary_emb.base == 42


def test_draft_mixin_defaults_rope_theta_to_10000_when_missing() -> None:
    """``getattr(draft_config, "rope_theta", 10000)`` fallback resolves to 10000."""

    class _NoBackendDraft(MobilintEagle3DraftModelMixin, nn.Module):
        def __init__(self, config, draft_config, fc_projector) -> None:
            nn.Module.__init__(self)
            self.draft_config = draft_config
            self.fc_projector = fc_projector
            self.embed_tokens = nn.Embedding(
                draft_config.vocab_size, draft_config.hidden_size, draft_config.pad_token_id
            )
            self.rotary_emb = self._build_draft_rotary_emb(draft_config)

    class _MinimalDraftConfig:
        vocab_size = 32
        hidden_size = 16
        num_attention_heads = 2
        pad_token_id = 0
        max_position_embeddings = 128

    draft = _NoBackendDraft(config=None, draft_config=_MinimalDraftConfig(), fc_projector=None)
    assert isinstance(draft.rotary_emb, CachedRotaryEmbedding)
    assert draft.rotary_emb.base == 10000


def test_draft_bookkeeping_fields_are_installed_by_mixin(stub_npu_backend: None) -> None:
    """Every draft bookkeeping attribute the concrete class used to set is present."""
    del stub_npu_backend
    config = _make_qwen2_config()
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = MobilintQwen2Eagle3DraftModel(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert draft.top_k == config.eagle3_tree_top_k
    assert draft.depth == config.eagle3_tree_depth
    assert draft.hidden_size == config.draft_config.hidden_size
    assert isinstance(draft.logsoftmax, nn.LogSoftmax)
    assert draft.d2t.shape == (config.draft_config.vocab_size,)
    assert draft.d2t.dtype == torch.long
    assert draft.t2d.shape == (config.draft_config.vocab_size,)
    assert draft.t2d.dtype == torch.bool
    assert draft.tree_mask_init.shape == (1, 1, draft.top_k, draft.top_k)
    assert draft.position_ids.shape == (draft.top_k,)
    for param in draft.embed_tokens.parameters():
        assert param.requires_grad is False
    assert draft.draft_config is config.draft_config
    assert draft.fc_projector is fc_projector
