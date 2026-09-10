"""Qwen3-VL vision output-order resolution.

The vision MXQ emits four tensors that all share one shape, so a wrong mapping
raises nothing -- it just degrades quality. These tests pin the resolution
order (env -> config field -> hardcoded default), env being process-wide for
local re-compile iteration and the config field being the release channel.

Lives under ``tests/transformers`` so ``scripts/test_transformers_matrix.py``
picks it up across the supported Transformers versions.
"""

import logging as _logging
from types import SimpleNamespace

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    DEFAULT_VISION_OUTPUT_ORDER,
    VISION_OUTPUT_ORDER_ENV,
    MobilintQwen3VLVisionModel,
)


def _model(*, vision_output_order=None):
    """Vision-model stub carrying only what order resolution touches: ``self.config``."""
    m = MobilintQwen3VLVisionModel.__new__(MobilintQwen3VLVisionModel)
    m.config = SimpleNamespace(vision_output_order=vision_output_order)
    return m


class TestResolution:
    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv(VISION_OUTPUT_ORDER_ENV, "3,0,1,2")
        m = _model(vision_output_order=[1, 0, 2, 3])
        assert m._resolve_vision_output_order() == (3, 0, 1, 2)

    def test_config_field_is_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(vision_output_order=[3, 0, 1, 2])
        assert m._resolve_vision_output_order() == (3, 0, 1, 2)

    def test_default_when_config_omits_field(self, monkeypatch):
        """A config predating this field (attribute missing / ``None``) keeps
        the shipped-encoder default -- backward compatibility."""
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(vision_output_order=None)
        assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER


class TestConfigLoudFailure:
    """``config.vision_output_order`` is a published release artifact.
    A broken value is a packaging bug and must surface loudly, not degrade
    quality silently the way a runtime-detected mismatch would."""

    @pytest.mark.parametrize(
        "bad",
        [
            [0, 1, 2],          # too short
            [0, 1, 2, 3, 4],    # too long
            [0, 1, 2, 2],       # not a permutation
            [1, 2, 3, 4],       # wrong domain
            ["a", "b", "c", "d"],  # non-integer
            3,                  # scalar (would raise TypeError from ``list(3)``)
            False,              # scalar (bool is int but ``list(False)`` raises)
            {"0": 3, "1": 0, "2": 1, "3": 2},  # dict: ``list(...)`` would give keys
            {0, 1, 2, 3},       # set: ``list(...)`` order is undefined
        ],
    )
    def test_bad_config_value_raises(self, monkeypatch, bad):
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(vision_output_order=bad)
        with pytest.raises(ValueError, match="config.vision_output_order"):
            m._resolve_vision_output_order()


class TestEnvLoudFailure:
    """Env is developer intent; hiding a typo would hide their mistake."""

    @pytest.mark.parametrize("bad", ["not,a,perm,ut", "0,1,2", "0,0,0,0", "1,2,3,4", "abc"])
    def test_bad_env_raises_at_resolve_time(self, monkeypatch, bad):
        monkeypatch.setenv(VISION_OUTPUT_ORDER_ENV, bad)
        m = _model()
        with pytest.raises(ValueError, match=VISION_OUTPUT_ORDER_ENV):
            m._resolve_vision_output_order()

    def test_empty_env_falls_back_silently(self, monkeypatch):
        """``export MBLT_VISION_OUTPUT_ORDER=`` must un-set, not raise."""
        monkeypatch.setenv(VISION_OUTPUT_ORDER_ENV, "")
        m = _model()
        assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER


class TestLogging:
    def test_default_load_is_not_info(self, monkeypatch, caplog):
        """The default path is the common case; it must not log at info."""
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model()
        with caplog.at_level(_logging.INFO):
            assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER
        assert not [r for r in caplog.records if r.levelno >= _logging.INFO]

    def test_config_load_logs_at_info(self, monkeypatch, caplog):
        """A config override must be visible in the run log so operators can
        confirm which mapping was applied."""
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(vision_output_order=[3, 0, 1, 2])
        with caplog.at_level(_logging.INFO):
            m._resolve_vision_output_order()
        text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= _logging.INFO)
        assert "config.vision_output_order" in text


class TestParser:
    @pytest.mark.parametrize("bad", [None, 3, 3.5])
    def test_non_iterable_raises_value_error_not_type_error(self, bad):
        with pytest.raises(ValueError):
            MobilintQwen3VLVisionModel._parse_vision_output_order(bad, "test")

    def test_accepts_string_list_and_tuple(self):
        f = MobilintQwen3VLVisionModel._parse_vision_output_order
        assert f("3, 0, 1, 2", "test") == (3, 0, 1, 2)
        assert f([3, 0, 1, 2], "test") == (3, 0, 1, 2)
        assert f((3, 0, 1, 2), "test") == (3, 0, 1, 2)

    def test_rejects_mapping_and_set(self):
        """Dicts and sets are iterables that would slip past a naive
        ``list(value)`` conversion -- dict keys and set order both mask the
        intended mapping. Reject them explicitly."""
        f = MobilintQwen3VLVisionModel._parse_vision_output_order
        with pytest.raises(ValueError, match="expected a list"):
            f({"0": 3, "1": 0, "2": 1, "3": 2}, "test")
        with pytest.raises(ValueError, match="expected a list"):
            f({0, 1, 2, 3}, "test")
