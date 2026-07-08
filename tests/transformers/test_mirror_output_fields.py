"""Regression tests for ``mirror_output_fields``.

The helper populates a target dataclass by mirroring values from a source
model output. When the source is itself a dataclass (the prod case — HF
``ModelOutput`` subclasses are dataclasses), only its declared fields are
eligible for mirroring so non-field attributes (``to_tuple``, ``keys``, ...)
whose names may collide with target fields in the future cannot silently
smuggle a wrong value through. Non-dataclass sources fall back to
``hasattr``/``getattr`` so tests can pass simple namespaces.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from mblt_model_zoo.hf_transformers.utils.generation_utils import (
    mirror_output_fields,
)


@dataclasses.dataclass
class _Target:
    last_hidden_state: object = None
    past_key_values: object = None
    keys: object = None


@dataclasses.dataclass
class _DataclassSource:
    last_hidden_state: object = None
    past_key_values: object = None


class TestMirrorOutputFieldsDataclassSource:
    def test_mirrors_declared_fields_only(self) -> None:
        source = _DataclassSource(last_hidden_state="hs", past_key_values="pkv")
        result = mirror_output_fields(_Target, source)
        assert result.last_hidden_state == "hs"
        assert result.past_key_values == "pkv"
        assert result.keys is None

    def test_non_field_attribute_collision_is_ignored(self) -> None:
        """Non-field attribute on source must not be mirrored to a target field."""
        source = _DataclassSource(last_hidden_state="hs", past_key_values="pkv")
        # Simulate a future upstream adding a non-dataclass attribute with a
        # name that happens to collide with a target field.
        source.keys = "SMUGGLED"  # type: ignore[attr-defined]

        result = mirror_output_fields(_Target, source)

        assert result.keys is None, (
            "mirror_output_fields must not pick up non-field attributes from "
            "a dataclass source even when the name matches a target field"
        )

    def test_overrides_win_over_source(self) -> None:
        source = _DataclassSource(last_hidden_state="hs", past_key_values="pkv")
        result = mirror_output_fields(_Target, source, past_key_values="override")
        assert result.past_key_values == "override"
        assert result.last_hidden_state == "hs"


class TestMirrorOutputFieldsDuckTypedSource:
    """SimpleNamespace fallback path — used by existing contract tests."""

    def test_hasattr_fallback_mirrors_present_attributes(self) -> None:
        source = SimpleNamespace(last_hidden_state="hs", past_key_values="pkv")
        result = mirror_output_fields(_Target, source)
        assert result.last_hidden_state == "hs"
        assert result.past_key_values == "pkv"

    def test_hasattr_fallback_ignores_missing_attributes(self) -> None:
        source = SimpleNamespace(last_hidden_state="hs")
        result = mirror_output_fields(_Target, source)
        assert result.last_hidden_state == "hs"
        assert result.past_key_values is None
