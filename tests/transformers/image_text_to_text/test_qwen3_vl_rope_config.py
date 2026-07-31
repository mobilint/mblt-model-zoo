"""Tests for Qwen3-VL rotary embedding config wiring across Transformers versions.

Transformers 5.x turned :class:`PreTrainedConfig` into a dataclass and folded the
legacy ``rope_theta`` field into ``rope_parameters`` (aliased via the
``rope_scaling`` property) during ``__post_init__``. Composite Qwen3-VL configs
built through that path expose ``rope_scaling["rope_theta"]`` but no longer
carry a top-level ``rope_theta`` attribute. Transformers 4.x still keeps
``rope_theta`` as its own attribute. :class:`MobilintQwen3VLRotaryEmbedding`
must accept both layouts.
"""

from __future__ import annotations

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLTextConfig,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLRotaryEmbedding,
)


def _minimal_text_config(**overrides) -> MobilintQwen3VLTextConfig:
    """Build a Qwen3-VL text config using arguments accepted by 4.x and 5.x."""
    kwargs = {
        "hidden_size": 4096,
        "head_dim": 128,
        "max_position_embeddings": 64,
        "rope_theta": 5_000_000,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "vocab_size": 151936,
    }
    kwargs.update(overrides)
    return MobilintQwen3VLTextConfig(**kwargs)


def test_rotary_embedding_reads_rope_theta_from_text_config() -> None:
    """Instantiate ``MobilintQwen3VLRotaryEmbedding`` from a Mobilint text config.

    Regression: Transformers 5.x drops the flat ``rope_theta`` attribute during
    ``PreTrainedConfig.__post_init__`` (folded into ``rope_parameters``). The
    rotary embedding module used to read ``config.rope_theta`` directly, which
    raised ``AttributeError`` on Transformers 5.x.
    """
    config = _minimal_text_config()

    emb = MobilintQwen3VLRotaryEmbedding(config)

    assert emb.rope_theta == 5_000_000
    assert emb.mrope_section == [24, 20, 20]


def test_rotary_embedding_falls_back_to_rope_scaling_when_flat_attr_missing() -> None:
    """Read ``rope_theta`` from ``rope_scaling`` when the flat attribute is absent.

    Simulates Transformers 5.x behavior where ``rope_theta`` is only present
    inside ``rope_parameters`` (aliased via the ``rope_scaling`` property).
    """
    config = _minimal_text_config()

    # Force the 5.x layout: rope_theta lives inside rope_scaling only. On 4.x
    # the class-level init keeps rope_theta as a flat attribute and does not
    # merge it into rope_scaling; on 5.x the dataclass __post_init__ already
    # merged it in, but we normalize here so the same test exercises both.
    scaling = dict(config.rope_scaling or {})
    scaling.setdefault("rope_theta", 5_000_000)
    config.rope_scaling = scaling

    if hasattr(config, "rope_theta"):
        try:
            delattr(config, "rope_theta")
        except AttributeError:
            config.__dict__.pop("rope_theta", None)
    assert not hasattr(config, "rope_theta")
    assert config.rope_scaling.get("rope_theta") == 5_000_000

    emb = MobilintQwen3VLRotaryEmbedding(config)

    assert emb.rope_theta == 5_000_000


def test_rotary_embedding_raises_when_rope_theta_missing_everywhere() -> None:
    """Preserve the hard-fail contract when the config truly has no rope_theta."""
    config = _minimal_text_config()

    # Strip rope_theta from both possible locations.
    if hasattr(config, "rope_theta"):
        try:
            delattr(config, "rope_theta")
        except AttributeError:
            config.__dict__.pop("rope_theta", None)
    rope_scaling = dict(config.rope_scaling)
    rope_scaling.pop("rope_theta", None)
    config.rope_scaling = rope_scaling

    with pytest.raises(ValueError, match="requires config.rope_theta"):
        MobilintQwen3VLRotaryEmbedding(config)
