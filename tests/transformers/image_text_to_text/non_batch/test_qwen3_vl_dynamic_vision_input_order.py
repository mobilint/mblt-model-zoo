"""Tests for auto-detection of the dynamic vision MXQ's 3-input slot order.

Two compile pipelines currently ship Qwen3-VL dynamic vision MXQs with
different input orderings and each is expected to load through the same
mblt-model-zoo wheel:

* mobilint's shipped 8B build (``mobilint/Qwen3-VL-8B-Instruct``) emits
  ``[rope, pos, folded]``.
* A tutorial recompile through ``VisionModelForQwen3VL.forward`` produces
  ``[folded, pos, rope]`` (qbcompiler places graph placeholders in
  dataflow order and ``folded`` is used first by ``patch_embed``).

``MobilintQwen3VLVisionModel._resolve_dynamic_input_slots`` recovers the
per-role slot index from the last-axis widths reported by the compiled
variant handle, so ``_prepare_dynamic_npu_inputs`` can hand the runtime
the tensors in the order the MXQ expects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLVisionModel,
)


def _vision_config_2b() -> SimpleNamespace:
    """Approximate ``MobilintQwen3VLVisionConfig`` fields for the shipped 2B build."""
    return SimpleNamespace(
        hidden_size=1024,
        num_heads=16,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=16,
    )


def _vision_config_8b() -> SimpleNamespace:
    """Approximate ``MobilintQwen3VLVisionConfig`` fields for the shipped 8B build."""
    return SimpleNamespace(
        hidden_size=1152,
        num_heads=16,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=16,
    )


@pytest.mark.parametrize(
    ("config_factory", "expected_widths"),
    [
        # rope = 2*alignUp(head_dim, 64); pos = hidden_size; folded = c*pt*ps^2
        (_vision_config_2b, (128, 1024, 1536)),  # head_dim=64 -> rope=128
        (_vision_config_8b, (256, 1152, 1536)),  # head_dim=72 -> rope=256
    ],
)
def test_resolve_slots_shipped_order(config_factory, expected_widths):
    """Shipped mobilint order ``[rope, pos, folded]`` maps to ``{r:0, p:1, f:2}``."""
    config = config_factory()
    rope_w, pos_w, folded_w = expected_widths
    input_shapes = [(1, -1, rope_w), (1, -1, pos_w), (1, -1, folded_w)]

    slots = MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(input_shapes, config)

    assert slots == {"rope": 0, "pos": 1, "folded": 2}


@pytest.mark.parametrize(
    ("config_factory", "expected_widths"),
    [
        (_vision_config_2b, (128, 1024, 1536)),
        (_vision_config_8b, (256, 1152, 1536)),
    ],
)
def test_resolve_slots_tutorial_order(config_factory, expected_widths):
    """Tutorial recompile order ``[folded, pos, rope]`` maps to ``{r:2, p:1, f:0}``."""
    config = config_factory()
    rope_w, pos_w, folded_w = expected_widths
    input_shapes = [(1, -1, folded_w), (1, -1, pos_w), (1, -1, rope_w)]

    slots = MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(input_shapes, config)

    assert slots == {"rope": 2, "pos": 1, "folded": 0}


def test_resolve_slots_permuted_order():
    """A permutation the compiler never emits still resolves as long as widths match."""
    config = _vision_config_2b()
    # arbitrary permutation [pos, folded, rope]
    input_shapes = [(1, -1, 1024), (1, -1, 1536), (1, -1, 128)]

    slots = MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(input_shapes, config)

    assert slots == {"rope": 2, "pos": 0, "folded": 1}


def test_resolve_slots_wrong_input_count_raises():
    """A 2-input (legacy) or N-input (unknown) shape list must not silently pass."""
    config = _vision_config_2b()

    with pytest.raises(ValueError, match="exactly 3 inputs"):
        MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(
            [(1, -1, 128), (1, -1, 1024)], config
        )


def test_resolve_slots_unexpected_width_raises():
    """Widths that don't match the config-derived rope/pos/folded triple raise."""
    config = _vision_config_2b()
    # Swap folded (1536) for an arbitrary bogus width
    input_shapes = [(1, -1, 128), (1, -1, 1024), (1, -1, 999)]

    with pytest.raises(ValueError, match="do not match the expected folded width"):
        MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(input_shapes, config)


def test_resolve_slots_role_width_collision_raises():
    """If config-derived widths collide we cannot recover the order — raise early."""

    class _CollidingConfig(SimpleNamespace):
        pass

    # Craft a synthetic (non-realistic) config where rope width equals pos width.
    # head_dim = 64 -> rope = 128; force hidden_size = 128 to collide.
    config = _CollidingConfig(
        hidden_size=128, num_heads=2, in_channels=3, temporal_patch_size=2, patch_size=16
    )
    input_shapes = [(1, -1, 128), (1, -1, 128), (1, -1, 1536)]

    with pytest.raises(ValueError, match="role widths collide"):
        MobilintQwen3VLVisionModel._resolve_dynamic_input_slots(input_shapes, config)


class _DispatchProbe(MobilintQwen3VLVisionModel):
    """Bypass NPU init; call ``_prepare_dynamic_npu_inputs`` on a fake."""

    def __init__(self, slots: dict[str, int]) -> None:
        import torch.nn as nn_

        nn_.Module.__init__(self)
        self._dynamic_input_slots = dict(slots)

    def compute_side_inputs(self, grid_thw):
        import torch

        n = int(grid_thw.prod().item())
        return torch.zeros(n, 1024), None

    def _build_vision_rotate_tensor(self, grid_thw):
        import numpy as np

        n = int(grid_thw.prod().item())
        # Fill each role's array with a unique constant so the assertions
        # below can identify which slot got which payload regardless of order.
        return np.full((1, n, 128), 3.0, dtype=np.float32)


def test_prepare_dynamic_npu_inputs_respects_resolved_slots():
    """Payloads land in the slot the resolver assigned, not a hard-coded index."""
    import numpy as np
    import torch

    # tutorial ordering: rope last
    probe = _DispatchProbe(slots={"rope": 2, "pos": 1, "folded": 0})

    n = 64
    fold_in = 1536
    hidden = torch.arange(n * fold_in, dtype=torch.float32).reshape(n, fold_in) + 1.0
    grid = torch.tensor([1, 8, 8])

    payloads = probe._prepare_dynamic_npu_inputs(hidden, grid)

    assert isinstance(payloads, list) and len(payloads) == 3
    # rope slot 2: sentinel value 3.0 from _build_vision_rotate_tensor
    assert np.all(payloads[2] == 3.0)
    assert payloads[2].shape[-1] == 128
    # pos slot 1: zeros from compute_side_inputs
    assert payloads[1].shape[-1] == 1024
    assert np.all(payloads[1] == 0.0)
    # folded slot 0: last axis width matches fold_in
    assert payloads[0].shape[-1] == fold_in


def test_prepare_dynamic_npu_inputs_shipped_slot_order():
    """Same input tensors, shipped ``[rope, pos, folded]`` slot layout."""
    import numpy as np
    import torch

    probe = _DispatchProbe(slots={"rope": 0, "pos": 1, "folded": 2})

    n = 64
    fold_in = 1536
    hidden = torch.arange(n * fold_in, dtype=torch.float32).reshape(n, fold_in) + 1.0
    grid = torch.tensor([1, 8, 8])

    payloads = probe._prepare_dynamic_npu_inputs(hidden, grid)

    assert payloads[0].shape[-1] == 128 and np.all(payloads[0] == 3.0)
    assert payloads[1].shape[-1] == 1024
    assert payloads[2].shape[-1] == fold_in
