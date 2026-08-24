"""Regression tests for bfloat16-safe RoPE table serialization.

The EAGLE-3 base model calls ``ScaledCachedRotaryEmbedding.forward`` with
``x.dtype`` = ``bfloat16`` when the release ships ``torch_dtype: bfloat16``
(for example Llama-3.1-8B). When the module was first constructed on
``meta`` device, the position table is deferred until the first forward,
where ``_build_position_table`` is called with the caller's dtype.
NumPy has no ``bfloat16`` dtype, so ``rotate_tensor.cpu().numpy()`` used
to raise ``TypeError: Got unsupported ScalarType BFloat16``.

These tests pin the fix: the packing tensor is cast to ``float32`` before
``.numpy()`` so the position table stays a float32 numpy array regardless
of the caller-supplied compute dtype. ``CachedRotaryEmbedding`` shares the
exact same code shape, so it is covered by parallel test cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from mblt_model_zoo.hf_transformers.utils.eagle3.eagle3_utils import (
    CachedRotaryEmbedding,
    ScaledCachedRotaryEmbedding,
)

_MAX_SEQ_LEN = 64
_HEAD_DIM = 32
# ``_build_position_table`` pads to a multiple of 64 half-channels, then stores
# ``[max_seq_len, 2 * padded_half_channels]``.
_PADDED_LAST_DIM = 2 * (((_HEAD_DIM + 63) // 64) * 64)


def _llama_config() -> LlamaConfig:
    """Build a small LlamaConfig with llama3 rope_scaling metadata attached.

    The full llama3 rope init pipeline in current Transformers is orthogonal
    to the bug under test — the failure is in packing-tensor serialization,
    not RoPE math. We therefore leave ``rope_type='default'`` on the module
    (so construction is portable across Transformers versions) while still
    exposing the llama3 rope_scaling metadata that the release ships.
    """
    return LlamaConfig(
        hidden_size=_HEAD_DIM * 4,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=64,
        max_position_embeddings=_MAX_SEQ_LEN,
        rope_theta=500_000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            # LlamaConfig enforces ``original_max_position_embeddings < max_position_embeddings``;
            # we keep the metadata shape but scale it to the test's tiny max_position_embeddings.
            "original_max_position_embeddings": _MAX_SEQ_LEN // 2,
        },
    )


def _fresh_scaled_module() -> ScaledCachedRotaryEmbedding:
    """Instantiate ScaledCachedRotaryEmbedding under the meta-init pattern.

    The real bug path is triggered when the module was materialized on meta
    (position_table stays ``None``) and the first ``forward`` call rebuilds
    the table with ``x.dtype``. We reproduce that by clearing the table
    manually so we can call ``_build_position_table(dtype=...)`` directly.

    Construction uses ``rope_type='default'`` for portability across
    Transformers versions (see docstring on ``_llama_config``); the packing
    tensor serialization under test is orthogonal to which RoPE init function
    populated ``inv_freq`` and ``attention_scaling``.
    """
    module = ScaledCachedRotaryEmbedding(
        dim=_HEAD_DIM,
        max_position_embeddings=_MAX_SEQ_LEN,
        rope_type="default",
    )
    # Attach the llama3 metadata so any future refactor that reads
    # ``module.config.rope_scaling`` inside ``_build_position_table`` still
    # exercises the Llama-3.1-8B code shape.
    module.config = _llama_config()
    module.position_table = None
    return module


def _fresh_cached_module() -> CachedRotaryEmbedding:
    module = CachedRotaryEmbedding(dim=_HEAD_DIM, max_position_embeddings=_MAX_SEQ_LEN)
    module.position_table = None
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_scaled_build_position_table_dtype(dtype: torch.dtype) -> None:
    """``_build_position_table`` returns fp32 numpy for any supported compute dtype."""
    module = _fresh_scaled_module()
    module._build_position_table(device=torch.device("cpu"), dtype=dtype)

    assert module.position_table is not None
    assert isinstance(module.position_table, np.ndarray)
    assert module.position_table.dtype == np.float32
    assert module.position_table.shape == (_MAX_SEQ_LEN, _PADDED_LAST_DIM)
    assert np.all(np.isfinite(module.position_table))


def test_scaled_forward_with_bf16_input_matches_llama_scenario() -> None:
    """End-to-end forward with bf16 input reproduces the Llama-3.1-8B scenario.

    The forward path used to blow up inside ``_build_position_table`` when the
    module was meta-initialized and the first live call arrived with a bf16
    ``x``. The regression guard: the returned numpy is fp32 and finite, and
    ``.astype(np.float32, copy=False)`` (as used downstream at
    ``mblt_model_zoo/hf_transformers/utils/eagle3/eagle3_utils.py:461``) is
    a no-op with no error.
    """
    module = _fresh_scaled_module()
    seq = 4
    x = torch.randn(1, seq, _HEAD_DIM * 4, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    result = module.forward(x, position_ids)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 1, seq, _PADDED_LAST_DIM)
    assert np.all(np.isfinite(result))
    # Downstream consumer path — must not error.
    result_astype = result.astype(np.float32, copy=False)
    assert result_astype is result or result_astype.dtype == np.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cached_build_position_table_dtype(dtype: torch.dtype) -> None:
    """``CachedRotaryEmbedding._build_position_table`` shares the same fix."""
    module = _fresh_cached_module()
    module._build_position_table(device=torch.device("cpu"), dtype=dtype)

    assert module.position_table is not None
    assert isinstance(module.position_table, np.ndarray)
    assert module.position_table.dtype == np.float32
    assert module.position_table.shape == (_MAX_SEQ_LEN, _PADDED_LAST_DIM)
    assert np.all(np.isfinite(module.position_table))


def test_cached_forward_with_bf16_input() -> None:
    """``CachedRotaryEmbedding.forward`` accepts bf16 x without crashing on ``.numpy()``."""
    module = _fresh_cached_module()
    seq = 4
    x = torch.randn(1, seq, _HEAD_DIM * 4, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    result = module.forward(x, position_ids)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 1, seq, _PADDED_LAST_DIM)
    assert np.all(np.isfinite(result))
