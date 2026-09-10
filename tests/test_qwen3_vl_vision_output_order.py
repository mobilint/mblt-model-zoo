"""Qwen3-VL vision output-order resolution.

The vision MXQ emits four tensors that all share one shape, so a wrong mapping
raises nothing -- it just degrades quality. These tests pin the resolution
order (env -> sidecar -> default), the sidecar being found next to the MXQ the
backend actually loaded, and malformed sidecars falling back instead of raising.
"""

import json
from types import SimpleNamespace

import pytest

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
    DEFAULT_VISION_OUTPUT_ORDER,
    VISION_OUTPUT_ORDER_ENV,
    VISION_OUTPUT_ORDER_FILENAME,
    MobilintQwen3VLVisionModel,
)


def _model(*, declared, resolved=None, resolver_raises=False):
    """A vision model stub carrying only what order resolution touches."""
    m = MobilintQwen3VLVisionModel.__new__(MobilintQwen3VLVisionModel)

    def check_model_path(path):
        if resolver_raises:
            raise OSError("hub unreachable")
        return resolved if resolved is not None else path

    m.npu_backend = SimpleNamespace(mxq_path=declared, check_model_path=check_model_path)
    return m


def _write_sidecar(directory, order):
    p = directory / VISION_OUTPUT_ORDER_FILENAME
    p.write_text(json.dumps({"output_order": order}), encoding="utf-8")
    return p


class TestResolution:
    def test_env_wins(self, monkeypatch, tmp_path):
        _write_sidecar(tmp_path, [1, 0, 2, 3])
        monkeypatch.setenv(VISION_OUTPUT_ORDER_ENV, "3,0,1,2")
        m = _model(declared="v.mxq", resolved=str(tmp_path / "v.mxq"))
        assert m._resolve_vision_output_order() == (3, 0, 1, 2)

    def test_sidecar_is_found_next_to_the_resolved_mxq(self, monkeypatch, tmp_path):
        # The declared path is repo-relative; only the resolved one leads to the
        # sidecar. Resolving against the cwd instead would silently use the default.
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        monkeypatch.chdir(tmp_path / "..")
        sub = tmp_path / "modeldir"
        sub.mkdir()
        _write_sidecar(sub, [3, 0, 1, 2])
        m = _model(declared="all_vision_dynamic.mxq", resolved=str(sub / "all_vision_dynamic.mxq"))
        assert m._resolve_vision_output_order() == (3, 0, 1, 2)

    def test_default_when_nothing_declares_an_order(self, monkeypatch, tmp_path):
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(declared="v.mxq", resolved=str(tmp_path / "v.mxq"))
        assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER

    def test_resolver_failure_does_not_break_loading(self, monkeypatch, tmp_path):
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        m = _model(declared="v.mxq", resolver_raises=True)
        assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER


class TestMalformedSidecarFallsBack:
    @pytest.mark.parametrize("bad", [None, 3, "nope", [0, 1, 2], [0, 1, 2, 2], {"a": 1}])
    def test_bad_values_use_the_default(self, monkeypatch, tmp_path, bad):
        monkeypatch.delenv(VISION_OUTPUT_ORDER_ENV, raising=False)
        _write_sidecar(tmp_path, bad)
        m = _model(declared="v.mxq", resolved=str(tmp_path / "v.mxq"))
        # Must not raise: a malformed sidecar is logged and ignored.
        assert m._resolve_vision_output_order() == DEFAULT_VISION_OUTPUT_ORDER


class TestParser:
    @pytest.mark.parametrize("bad", [None, 3, 3.5])
    def test_non_iterable_raises_value_error_not_type_error(self, bad):
        with pytest.raises(ValueError):
            MobilintQwen3VLVisionModel._parse_vision_output_order(bad, "test")

    def test_accepts_string_and_sequence(self):
        f = MobilintQwen3VLVisionModel._parse_vision_output_order
        assert f("3, 0, 1, 2", "test") == (3, 0, 1, 2)
        assert f([3, 0, 1, 2], "test") == (3, 0, 1, 2)
