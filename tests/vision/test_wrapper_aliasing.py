"""Tests for vision model name aliasing."""

from __future__ import annotations

import pytest

from mblt_model_zoo.vision.wrapper import MBLT_Engine


@pytest.mark.parametrize(
    ("model_name", "expected_yaml"),
    [
        ("regnet_x_16gf", "RegNet_X_16GF.yaml"),
        ("regnet-x-16gf", "RegNet_X_16GF.yaml"),
        ("RegNet_X_16GF.yaml", "RegNet_X_16GF.yaml"),
        ("regnet_x_1_6gf", "RegNet_X_1_6GF.yaml"),
        ("regnet-x-1-6gf", "RegNet_X_1_6GF.yaml"),
        ("resnet50", "ResNet50.yaml"),
        ("resnet-50", "ResNet50.yaml"),
        ("resnet_50", "ResNet50.yaml"),
    ],
)
def test_model_name_aliasing_resolves_precise_separator_matches(
    model_name: str,
    expected_yaml: str,
) -> None:
    """Resolve aliases without collapsing distinct separator boundaries too early."""

    engine = MBLT_Engine.__new__(MBLT_Engine)

    assert engine.model_name_aliasing(model_name) == expected_yaml


def test_model_name_aliasing_reports_compact_ambiguity() -> None:
    """Keep compact ambiguous names explicit."""

    engine = MBLT_Engine.__new__(MBLT_Engine)

    with pytest.raises(ValueError, match="ambiguous"):
        engine.model_name_aliasing("regnetx16gf")
