"""Regression tests: caller cannot disable ``return_mm_token_type_ids`` on tf 5.x.

tf 5.x's ``Qwen3VLModel.compute_3d_position_ids`` and the generate-side
``_prepare_position_ids_for_generation`` build MRoPE 3-D t/h/w positions only
when ``mm_token_type_ids`` is present in the tokenizer output. Without it,
both fall back to linear (non-MRoPE) positions and the decoder cannot
distinguish visual tokens by time/space, producing degenerate output on video
inputs and stale position math on multi-image inputs.

Prior implementation used ``text_kwargs.setdefault('return_mm_token_type_ids',
True)``, so a caller-supplied ``False`` silently disabled MRoPE. The safety
envelope now forces ``True`` on multimodal runs by explicit assignment and
logs the overwrite.
"""

from __future__ import annotations

import logging

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402

from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLProcessor,
    MobilintQwen3VLVideoProcessor,
)


_PROCESSOR_LOGGER = "mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl"


class _ImageProcessorStub:
    """Minimal image processor with the attributes the safety envelope reads."""

    def __init__(self) -> None:
        self.size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        self.patch_size = 14
        self.merge_size = 2
        self.temporal_patch_size = 2


def _make_processor(dynamic_vision: bool = True) -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``."""
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = _ImageProcessorStub()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


def _skip_if_pre_mrope() -> None:
    """The MRoPE metadata invariant only applies on tf 5.x."""
    if not hasattr(MobilintQwen3VLProcessor, "create_mm_token_type_ids"):
        pytest.skip("tf 4.x has no ``create_mm_token_type_ids`` — invariant is inert")


def test_call_forces_return_mm_token_type_ids_true_over_caller_false(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller-supplied ``return_mm_token_type_ids=False`` is overwritten to True.

    This is the exact bypass the reviewer flagged: ``setdefault`` respected a
    caller False and silently disabled MRoPE on tf 5.x. The safety envelope
    now uses explicit assignment.
    """
    _skip_if_pre_mrope()

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["text_kwargs"] = kwargs.get("text_kwargs")
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=True)
    with caplog.at_level(logging.DEBUG, logger=_PROCESSOR_LOGGER):
        result = proc(
            images=[object()],
            text="describe <|image_pad|>",
            text_kwargs={"return_mm_token_type_ids": False},
        )

    assert result == "sentinel"
    forwarded_text_kwargs = captured["text_kwargs"]
    assert isinstance(forwarded_text_kwargs, dict)
    assert forwarded_text_kwargs["return_mm_token_type_ids"] is True
    assert any(
        "return_mm_token_type_ids" in rec.message and "-> True" in rec.message
        for rec in caplog.records
    ), f"expected an overwrite debug log, got: {[rec.message for rec in caplog.records]}"


def test_call_forces_return_mm_token_type_ids_true_over_caller_false_video(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Video path also overwrites a caller False. MRoPE is exactly what video needs."""
    _skip_if_pre_mrope()

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["text_kwargs"] = kwargs.get("text_kwargs")
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=True)
    with caplog.at_level(logging.DEBUG, logger=_PROCESSOR_LOGGER):
        result = proc(
            images=None,
            text="describe <|video_pad|>",
            videos=[object()],
            text_kwargs={"return_mm_token_type_ids": False},
        )

    assert result == "sentinel"
    forwarded_text_kwargs = captured["text_kwargs"]
    assert forwarded_text_kwargs["return_mm_token_type_ids"] is True
    assert any(
        "return_mm_token_type_ids" in rec.message and "-> True" in rec.message
        for rec in caplog.records
    )


def test_call_no_overwrite_log_when_caller_omits_return_mm_token_type_ids(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No debug log when the caller supplied no value — the default path is silent.

    The overwrite log is a warning-shaped signal for callers deliberately
    passing False; the omitted-kwarg default path is the common case and must
    not be noisy.
    """
    _skip_if_pre_mrope()

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["text_kwargs"] = kwargs.get("text_kwargs")
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=True)
    with caplog.at_level(logging.DEBUG, logger=_PROCESSOR_LOGGER):
        result = proc(images=[object()], text="describe <|image_pad|>")

    assert result == "sentinel"
    assert captured["text_kwargs"]["return_mm_token_type_ids"] is True
    assert not any(
        "overwriting caller-supplied" in rec.message for rec in caplog.records
    ), "no overwrite log expected when caller omitted the field"


def test_call_no_overwrite_log_when_caller_supplies_true(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller-supplied True is a no-op — no overwrite log."""
    _skip_if_pre_mrope()

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["text_kwargs"] = kwargs.get("text_kwargs")
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=True)
    with caplog.at_level(logging.DEBUG, logger=_PROCESSOR_LOGGER):
        result = proc(
            images=[object()],
            text="describe <|image_pad|>",
            text_kwargs={"return_mm_token_type_ids": True},
        )

    assert result == "sentinel"
    assert captured["text_kwargs"]["return_mm_token_type_ids"] is True
    assert not any(
        "overwriting caller-supplied" in rec.message for rec in caplog.records
    )


def test_call_text_only_leaves_return_mm_token_type_ids_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure text call (no images, no videos) does not force the field.

    The invariant only applies on multimodal runs — the field is meaningful
    only when there are vision tokens whose positions MRoPE has to lay out.
    Forcing it on text-only calls would leak the flag into upstream text
    tokenization for no benefit.
    """
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["text_kwargs"] = kwargs.get("text_kwargs")
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=True)
    result = proc(images=None, text="pure text prompt")

    assert result == "sentinel"
    # Envelope short-circuits: no text_kwargs was created for a text-only call.
    assert captured["text_kwargs"] is None


def test_apply_safety_envelope_overwrites_false_directly() -> None:
    """Direct method call: envelope replaces caller-supplied False in place."""
    _skip_if_pre_mrope()

    proc = _make_processor(dynamic_vision=True)
    kwargs: dict = {"text_kwargs": {"return_mm_token_type_ids": False}}

    proc._apply_safety_envelope(images=[object()], videos=None, kwargs=kwargs)

    assert kwargs["text_kwargs"]["return_mm_token_type_ids"] is True


def test_apply_safety_envelope_creates_text_kwargs_when_missing() -> None:
    """Direct method call: envelope creates a text_kwargs dict when caller omitted it."""
    _skip_if_pre_mrope()

    proc = _make_processor(dynamic_vision=True)
    kwargs: dict = {}

    proc._apply_safety_envelope(images=[object()], videos=None, kwargs=kwargs)

    assert kwargs["text_kwargs"]["return_mm_token_type_ids"] is True
