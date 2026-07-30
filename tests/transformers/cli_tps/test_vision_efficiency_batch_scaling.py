"""Regression tests for VLM vision energy-efficiency metrics.

`measure_vision(..., batch_size=N)` processes ``N`` images per invocation while
the energy tracker window integrates over the entire call, so the raw joules
figure covers the whole batch. The vision efficiency pair must therefore be
scaled by ``batch_size``:

    vision_img_per_j = batch_size / energy
    vision_j_per_img = energy / batch_size

This test guards against a regression where the scaling factor was 1 instead
of ``batch_size`` (Codex Review flag on PR #102).
"""

import pytest

from mblt_model_zoo.cli import tps as tps_cli


def test_vision_efficiency_scales_by_batch_size() -> None:
    img_per_j, j_per_img = tps_cli._vision_efficiency_metrics([8.0], batch_size=4)
    assert img_per_j == [0.5]
    assert j_per_img == [2.0]


def test_vision_efficiency_metrics_are_reciprocals_per_image() -> None:
    img_per_j, j_per_img = tps_cli._vision_efficiency_metrics([8.0, 5.0], batch_size=4)
    for a, b in zip(img_per_j, j_per_img):
        assert a * b == pytest.approx(1.0)


def test_vision_efficiency_skips_nonpositive_energy_for_img_per_j() -> None:
    img_per_j, j_per_img = tps_cli._vision_efficiency_metrics([8.0, 0.0, -1.0], batch_size=4)
    assert img_per_j == [0.5]
    assert j_per_img == [2.0, 0.0, -0.25]


def test_vision_efficiency_batch_size_one_matches_raw_energy() -> None:
    img_per_j, j_per_img = tps_cli._vision_efficiency_metrics([2.0, 4.0], batch_size=1)
    assert img_per_j == [0.5, 0.25]
    assert j_per_img == [2.0, 4.0]
