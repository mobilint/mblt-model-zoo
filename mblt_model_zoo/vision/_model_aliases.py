"""Model configuration aliases for the vision YAML registry."""

from __future__ import annotations

from pathlib import Path

MODEL_CONFIG_ALIASES: dict[str, str] = {
    "DeiT3BasePatch16384": "DeiT3_Base_Patch16_384",
    "DeiT3LargePatch16224": "DeiT3_Large_Patch16_224",
    "DeiT3LargePatch16384": "DeiT3_Large_Patch16_384",
    "DeiT3MediumPatch16224": "DeiT3_Medium_Patch16_224",
    "DeiT3SmallPatch16224": "DeiT3_Small_Patch16_224",
    "DeiT3SmallPatch16384": "DeiT3_Small_Patch16_384",
    "DeiTBasePatch16384": "DeiT_Base_Patch16_384",
    "DeiTSmallPatch16224": "DeiT_Small_Patch16_224",
    "DeiTTinyPatch16224": "DeiT_Tiny_Patch16_224",
    "EfficientNetB1": "EfficientNet_B1",
    "InceptionV3": "Inception_V3",
    "ViTB16": "ViT_B_16",
    "ViTL16": "ViT_L_16",
    "WideResNet1012": "Wide_ResNet101_2",
    "WideResNet502": "Wide_ResNet50_2",
}

_NORMALIZED_MODEL_CONFIG_ALIASES = {alias.lower(): canonical for alias, canonical in MODEL_CONFIG_ALIASES.items()}


def strip_yaml_suffix(model_name: str) -> str:
    """Returns a model name without a trailing YAML suffix."""

    return model_name[: -len(".yaml")] if model_name.lower().endswith(".yaml") else model_name


def resolve_model_config_alias(model_name: str) -> str | None:
    """Returns the canonical YAML stem for a known legacy model config alias."""

    model_stem = Path(strip_yaml_suffix(model_name)).name
    return _NORMALIZED_MODEL_CONFIG_ALIASES.get(model_stem.lower())
