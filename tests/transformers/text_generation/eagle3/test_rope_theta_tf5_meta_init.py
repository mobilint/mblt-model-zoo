"""Regression tests for TF 5.x rope_theta lookup and meta-init re-init contract.

Two independent bugs cause EAGLE-3 draft-model quality to silently degrade on
Transformers 5.x:

1. TF 5.x turned :class:`PretrainedConfig` into a dataclass and folded top-level
   ``rope_theta`` into ``rope_parameters`` (exposed via the ``rope_scaling``
   property). Plain ``getattr(config, "rope_theta", 10000)`` silently falls back
   to ``10000`` and both base and draft RoPE tables use the wrong theta.
2. TF 5.x's :meth:`PreTrainedModel._init_weights` has a re-init branch for
   ``RotaryEmbedding``-like modules, gated on ``hasattr(module, "original_inv_freq")``
   and a ``compute_default_rope_parameters``/``rope_type``/``config`` triple.
   Without those, ``inv_freq`` is left as uninitialized memory after the meta
   materialization pass and the draft speaks noise.

These tests exercise the fix on any Transformers version by simulating the
TF 5.x wire form (``rope_theta`` inside ``rope_scaling``) directly, and by
constructing the rotary modules under :class:`torch.device("meta")` to
reproduce the meta-materialization scenario.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from mblt_model_zoo.hf_transformers.models.llama_eagle3.configuration_llama_eagle3 import (
    MobilintLlamaEagle3Config,
)
from mblt_model_zoo.hf_transformers.models.llama_eagle3.modeling_llama_eagle3 import (
    MobilintLlamaEagle3BaseModel,
    MobilintLlamaEagle3DraftModel,
)
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
    MobilintEagle3FCProjector,
    ScaledCachedRotaryEmbedding,
    _resolve_rope_theta,
)
from mblt_model_zoo.utils.npu_backend import MobilintNPUBackend

_TRUTHY_DEFAULT_ROPE_SCALING: dict[str, Any] = {
    "rope_type": "default",
    "factor": 1.0,
}


@pytest.fixture
def stub_npu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``MobilintNPUBackend.create`` / ``launch`` so the mixin chain runs offline."""
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: None)
    monkeypatch.setattr(MobilintNPUBackend, "launch", lambda self: None)
    return None


class _TF5StyleConfig:
    """Minimal duck-typed config that simulates the TF 5.x wire form.

    TF 5.x drops the flat ``rope_theta`` attribute and folds the value into
    ``rope_parameters`` / ``rope_scaling`` during ``__post_init__``. This
    fixture reproduces that layout without depending on the installed TF
    version, so the regression can be exercised on TF 4.x too.
    """

    def __init__(self, *, rope_theta: float, extra_scaling: dict[str, Any] | None = None) -> None:
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.max_position_embeddings = 128
        self.pad_token_id = 0
        self.vocab_size = 64
        self.rope_scaling: dict[str, Any] = {"rope_type": "default", "rope_theta": rope_theta}
        if extra_scaling is not None:
            self.rope_scaling.update(extra_scaling)


def test_resolve_rope_theta_reads_top_level_attr() -> None:
    """TF 4.x layout: ``rope_theta`` is a flat attribute."""

    class _Cfg:
        rope_theta = 500_000.0
        rope_scaling = None

    assert _resolve_rope_theta(_Cfg()) == 500_000.0


def test_resolve_rope_theta_reads_rope_scaling_key() -> None:
    """TF 5.x layout: ``rope_theta`` lives inside ``rope_scaling`` / ``rope_parameters``."""

    class _Cfg:
        rope_scaling = {"rope_type": "default", "rope_theta": 1_000_000.0}

    assert _resolve_rope_theta(_Cfg()) == 1_000_000.0


def test_resolve_rope_theta_prefers_top_level_over_scaling() -> None:
    """When both layouts carry a value the flat attribute wins (TF 4.x source of truth)."""

    class _Cfg:
        rope_theta = 100.0
        rope_scaling = {"rope_theta": 200.0}

    assert _resolve_rope_theta(_Cfg()) == 100.0


def test_resolve_rope_theta_uses_default_when_unresolvable() -> None:
    """Absent both paths, fall back to the caller-provided default."""

    class _Cfg:
        rope_scaling = None

    assert _resolve_rope_theta(_Cfg(), default=42.0) == 42.0


def test_resolve_rope_theta_raises_when_neither_path_resolves() -> None:
    """No default + no theta anywhere → :class:`ValueError` that names both paths."""

    class _Cfg:
        rope_scaling = None

    with pytest.raises(ValueError, match="rope_theta"):
        _resolve_rope_theta(_Cfg())


def test_resolve_rope_theta_none_config_requires_default() -> None:
    """``config=None`` still requires an explicit default."""
    assert _resolve_rope_theta(None, default=7.0) == 7.0
    with pytest.raises(ValueError):
        _resolve_rope_theta(None)


def test_cached_rotary_resolves_tf5_style_rope_theta() -> None:
    """The draft rotary reads TF 5.x-style ``rope_theta`` from ``rope_scaling``.

    ``CachedRotaryEmbedding.compute_default_rope_parameters`` is the HF re-init
    hook. On TF 5.x it is called as ``rope_fn(module.config)`` and must resolve
    ``rope_theta`` from ``config.rope_scaling`` (not the missing flat attribute).
    """
    cfg = _TF5StyleConfig(rope_theta=500_000.0)
    module = CachedRotaryEmbedding(dim=16, max_position_embeddings=64, base=10000, config=cfg)

    # base= is a caller default; compute_default_rope_parameters overrides from config.
    inv_freq, scaling = module.compute_default_rope_parameters(cfg)
    reference = 1.0 / (500_000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16.0))
    assert scaling == pytest.approx(1.0)
    torch.testing.assert_close(inv_freq, reference)


def test_scaled_rotary_resolves_tf5_style_rope_theta() -> None:
    """``ScaledCachedRotaryEmbedding`` picks up the TF 5.x ``rope_theta`` too.

    The bug: at construction time ``self.base`` used to stay at the caller
    default (10000) because ``getattr(config, "rope_theta", ...)`` returned the
    fallback on TF 5.x. Now the helper resolves from ``rope_scaling`` so
    ``self.base`` matches the config.
    """
    cfg = _TF5StyleConfig(rope_theta=500_000.0)
    module = ScaledCachedRotaryEmbedding(
        dim=16,
        max_position_embeddings=64,
        base=10000,
        config=cfg,
    )
    assert module.base == 500_000

    # Also verify the re-init hook resolves from the same source.
    inv_freq, _ = module.compute_default_rope_parameters(cfg)
    reference = 1.0 / (500_000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16.0))
    torch.testing.assert_close(inv_freq, reference)


def test_scaled_rotary_position_table_uses_tf5_theta() -> None:
    """End-to-end: the pre-built position table reflects the config's theta on TF 5.x."""
    theta = 500_000.0
    cfg = _TF5StyleConfig(rope_theta=theta)
    module = ScaledCachedRotaryEmbedding(dim=16, max_position_embeddings=64, config=cfg)
    # Force rebuild in case __init__ used a stale inv_freq (regression guard).
    module.position_table = None
    module._build_position_table(device=torch.device("cpu"), dtype=torch.float32)

    # Reference table built directly with the right theta. Match max_seq_len to
    # what the config-driven path picks (config.max_position_embeddings wins).
    reference_module = ScaledCachedRotaryEmbedding(
        dim=16,
        max_position_embeddings=cfg.max_position_embeddings,
        base=int(theta),
        rope_type="default",
    )
    assert module.position_table is not None
    assert reference_module.position_table is not None
    np.testing.assert_allclose(module.position_table, reference_module.position_table, atol=1e-5)


def test_cached_rotary_meta_init_recovers_via_compute_default() -> None:
    """Meta-init + HF re-init: ``inv_freq`` recovers the correct values.

    Reproduces the TF 5.x meta-init trap. Under :class:`torch.device("meta")`
    the ``torch.arange`` inside :meth:`CachedRotaryEmbedding.__init__` yields
    a meta tensor and HF later materializes the buffer with uninitialized
    bytes. The re-init branch calls ``compute_default_rope_parameters(config)``
    to overwrite those bytes with correct values.
    """
    cfg = _TF5StyleConfig(rope_theta=500_000.0)
    with torch.device("meta"):
        module = CachedRotaryEmbedding(dim=16, max_position_embeddings=64, base=10000, config=cfg)

    # Sanity: the re-init contract requires all of these to be present.
    assert hasattr(module, "original_inv_freq")
    assert module.rope_type == "default"
    assert module.config is cfg
    assert callable(getattr(module, "compute_default_rope_parameters", None))

    # HF's re-init calls rope_fn(module.config) — reproduce that call directly.
    buffer_value, _ = module.compute_default_rope_parameters(module.config)
    assert buffer_value.device.type == "cpu"
    assert torch.all(torch.isfinite(buffer_value))
    reference = 1.0 / (500_000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16.0))
    torch.testing.assert_close(buffer_value, reference)


def test_cached_rotary_meta_init_provides_original_inv_freq_and_rope_type() -> None:
    """Meta-init CachedRotaryEmbedding still exposes the HF re-init hooks.

    Without ``original_inv_freq`` and ``rope_type`` the HF re-init branch skips
    the module and ``inv_freq`` stays as uninitialized memory.
    """
    with torch.device("meta"):
        module = CachedRotaryEmbedding(dim=16, max_position_embeddings=64, base=10000)
    assert "RotaryEmbedding" in type(module).__name__
    assert hasattr(module, "original_inv_freq")
    assert module.rope_type == "default"


def _make_qwen2_eagle3_config(*, tf5_style: bool, rope_theta: float | None) -> MobilintQwen2Eagle3Config:
    draft_kwargs: dict[str, Any] = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
    if rope_theta is not None:
        draft_kwargs["rope_theta"] = rope_theta

    cfg = MobilintQwen2Eagle3Config(
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
        name_or_path="mobilint/test-qwen2-eagle3-tf5",
    )
    if tf5_style and rope_theta is not None:
        # Simulate the TF 5.x __post_init__ fold: rope_theta moves into rope_scaling
        # and the flat attribute goes away.
        cfg.draft_config.rope_scaling = {"rope_type": "default", "rope_theta": rope_theta}
        if hasattr(cfg.draft_config, "rope_theta"):
            try:
                del cfg.draft_config.rope_theta
            except AttributeError:
                pass
    return cfg


def _make_qwen3_eagle3_config(*, tf5_style: bool, rope_theta: float | None) -> MobilintQwen3Eagle3Config:
    draft_kwargs: dict[str, Any] = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
    if rope_theta is not None:
        draft_kwargs["rope_theta"] = rope_theta

    cfg = MobilintQwen3Eagle3Config(
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
        name_or_path="mobilint/test-qwen3-eagle3-tf5",
    )
    if tf5_style and rope_theta is not None:
        cfg.draft_config.rope_scaling = {"rope_type": "default", "rope_theta": rope_theta}
        if hasattr(cfg.draft_config, "rope_theta"):
            try:
                del cfg.draft_config.rope_theta
            except AttributeError:
                pass
    return cfg


def _make_llama_eagle3_config(*, tf5_style: bool, rope_theta: float | None) -> MobilintLlamaEagle3Config:
    draft_kwargs: dict[str, Any] = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
    if rope_theta is not None:
        draft_kwargs["rope_theta"] = rope_theta

    cfg = MobilintLlamaEagle3Config(
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
        name_or_path="mobilint/test-llama-eagle3-tf5",
    )
    if tf5_style and rope_theta is not None:
        cfg.draft_config.rope_scaling = {"rope_type": "default", "rope_theta": rope_theta}
        if hasattr(cfg.draft_config, "rope_theta"):
            try:
                del cfg.draft_config.rope_theta
            except AttributeError:
                pass
    return cfg


_FAMILIES = [
    pytest.param(
        _make_qwen2_eagle3_config,
        MobilintQwen2Eagle3BaseModel,
        MobilintQwen2Eagle3DraftModel,
        id="qwen2_eagle3",
    ),
    pytest.param(
        _make_qwen3_eagle3_config,
        MobilintQwen3Eagle3BaseModel,
        MobilintQwen3Eagle3DraftModel,
        id="qwen3_eagle3",
    ),
    pytest.param(
        _make_llama_eagle3_config,
        MobilintLlamaEagle3BaseModel,
        MobilintLlamaEagle3DraftModel,
        id="llama_eagle3",
    ),
]


@pytest.mark.parametrize("config_factory, base_cls, draft_cls", _FAMILIES)
def test_draft_rope_theta_from_tf5_style_rope_scaling(
    stub_npu_backend: None,
    config_factory,
    base_cls,
    draft_cls,
) -> None:
    """Draft ``CachedRotaryEmbedding.base`` resolves from ``rope_scaling`` on TF 5.x."""
    del stub_npu_backend, base_cls
    config = config_factory(tf5_style=True, rope_theta=500_000.0)
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    draft = draft_cls(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )
    assert isinstance(draft.rotary_emb, CachedRotaryEmbedding)
    assert draft.rotary_emb.base == 500_000


@pytest.mark.parametrize("config_factory, base_cls, draft_cls", _FAMILIES)
def test_base_rope_theta_from_tf5_style_rope_scaling(
    stub_npu_backend: None,
    config_factory,
    base_cls,
    draft_cls,
) -> None:
    """Base ``ScaledCachedRotaryEmbedding.base`` resolves from ``rope_scaling`` on TF 5.x.

    Also simulate the TF 5.x fold on the top-level (base) config, so the
    scaled-rotary construction sees the wire form we care about.
    """
    del stub_npu_backend, draft_cls
    config = config_factory(tf5_style=True, rope_theta=500_000.0)
    # Simulate TF 5.x fold on the top-level config too.
    config.rope_scaling = {"rope_type": "default", "rope_theta": 750_000.0}
    if hasattr(config, "rope_theta"):
        try:
            del config.rope_theta
        except AttributeError:
            pass

    base = base_cls(config, _internal_call=True, no_launch=True)
    assert isinstance(base.rotary_emb, ScaledCachedRotaryEmbedding)
    assert base.rotary_emb.base == 750_000


class _NoopBaseWrapper(nn.Module):
    """Standalone base-mixin holder for meta-init testing without the NPU chain."""

    def __init__(self, base_mixin_cls, config: Any) -> None:
        super().__init__()
        base_mixin_cls.__init__(self, config)


@pytest.mark.parametrize("config_factory, base_cls, draft_cls", _FAMILIES)
def test_draft_meta_init_inv_freq_recovers(
    stub_npu_backend: None,
    config_factory,
    base_cls,
    draft_cls,
) -> None:
    """Meta-init draft rotary + HF re-init contract: inv_freq matches the non-meta reference.

    The re-init hook (``compute_default_rope_parameters``) is what HF calls on
    TF 5.x to overwrite the uninitialized bytes. Here we invoke it directly and
    check the values against a fresh non-meta construction.
    """
    del stub_npu_backend, base_cls
    theta = 500_000.0
    config = config_factory(tf5_style=True, rope_theta=theta)

    # Reference: construct the draft off meta, so inv_freq is populated correctly.
    fc_projector = MobilintEagle3FCProjector(config, _internal_call=True, no_launch=True)
    reference = draft_cls(
        config,
        draft_config=config.draft_config,
        fc_projector=fc_projector,
        _internal_call=True,
        no_launch=True,
    )

    with torch.device("meta"):
        meta_config = config_factory(tf5_style=True, rope_theta=theta)
        meta_fc = MobilintEagle3FCProjector(meta_config, _internal_call=True, no_launch=True)
        meta_draft = draft_cls(
            meta_config,
            draft_config=meta_config.draft_config,
            fc_projector=meta_fc,
            _internal_call=True,
            no_launch=True,
        )

    # HF re-init contract: name check + ``original_inv_freq`` + ``rope_type`` +
    # ``compute_default_rope_parameters`` + ``config``.
    module = meta_draft.rotary_emb
    assert isinstance(module, CachedRotaryEmbedding)
    assert "RotaryEmbedding" in type(module).__name__
    assert hasattr(module, "original_inv_freq")
    assert module.rope_type == "default"
    assert module.config is meta_draft.draft_config

    # Simulate the HF re-init call.
    recovered_inv_freq, _ = module.compute_default_rope_parameters(module.config)
    torch.testing.assert_close(recovered_inv_freq, reference.rotary_emb.inv_freq)


@pytest.mark.parametrize("config_factory, base_cls, draft_cls", _FAMILIES)
def test_base_meta_init_inv_freq_recovers(
    stub_npu_backend: None,
    config_factory,
    base_cls,
    draft_cls,
) -> None:
    """Meta-init base rotary + HF re-init contract: inv_freq matches the non-meta reference.

    Bug 1 (rope_theta) previously left the base RoPE table on the wrong theta
    silently. With the helper wired into the default rope path, the recovered
    ``inv_freq`` matches a freshly-built non-meta reference exactly.
    """
    del stub_npu_backend, draft_cls
    theta = 500_000.0
    config = config_factory(tf5_style=True, rope_theta=theta)
    config.rope_scaling = {"rope_type": "default", "rope_theta": 750_000.0}
    if hasattr(config, "rope_theta"):
        try:
            del config.rope_theta
        except AttributeError:
            pass

    reference = base_cls(config, _internal_call=True, no_launch=True)

    with torch.device("meta"):
        meta_config = config_factory(tf5_style=True, rope_theta=theta)
        meta_config.rope_scaling = {"rope_type": "default", "rope_theta": 750_000.0}
        if hasattr(meta_config, "rope_theta"):
            try:
                del meta_config.rope_theta
            except AttributeError:
                pass
        meta_base = base_cls(meta_config, _internal_call=True, no_launch=True)

    module = meta_base.rotary_emb
    assert isinstance(module, ScaledCachedRotaryEmbedding)
    assert "RotaryEmbedding" in type(module).__name__
    assert hasattr(module, "original_inv_freq")
    assert module.rope_type == "default"
    assert module.config is meta_config

    recovered_inv_freq, _ = module.compute_default_rope_parameters(module.config)
    torch.testing.assert_close(recovered_inv_freq, reference.rotary_emb.inv_freq)
