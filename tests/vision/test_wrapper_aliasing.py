"""Tests for vision model name aliasing."""

from __future__ import annotations

import pytest

from mblt_model_zoo.vision.wrapper import MODEL_CONFIG_DIR, MBLT_Engine


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


def test_legacy_model_config_aliases_do_not_keep_duplicate_yaml_files() -> None:
    """Keep removed legacy config names out of the YAML registry."""

    duplicate_config_names = [
        "DeiT3BasePatch16384.yaml",
        "DeiT3LargePatch16224.yaml",
        "DeiT3LargePatch16384.yaml",
        "DeiT3MediumPatch16224.yaml",
        "DeiT3SmallPatch16224.yaml",
        "DeiT3SmallPatch16384.yaml",
        "DeiTBasePatch16384.yaml",
        "DeiTSmallPatch16224.yaml",
        "DeiTTinyPatch16224.yaml",
        "EfficientNetB1.yaml",
        "InceptionV3.yaml",
        "ViTB16.yaml",
        "ViTL16.yaml",
        "WideResNet1012.yaml",
        "WideResNet502.yaml",
    ]

    for config_name in duplicate_config_names:
        assert not (MODEL_CONFIG_DIR / config_name).exists()
