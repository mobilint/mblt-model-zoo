"""Tests for the Qwen3-VL ``dynamic_vision`` config field and its downstream wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402

from mblt_model_zoo.hf_transformers.models.qwen3_vl import processing_qwen3_vl  # noqa: E402
from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLConfig,
    MobilintQwen3VLVisionConfig,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLModel,
    MobilintQwen3VLVisionModel,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    _NPU_MAX_VISION_TOKENS,
    MobilintQwen3VLProcessor,
    MobilintQwen3VLVideoProcessor,
)


class _VisionStub:
    def __init__(self, uses_dynamic_vision: bool) -> None:
        self._uses_dynamic_vision = uses_dynamic_vision


class _ModelStub:
    """Mimic the Qwen3-VL model attribute layout expected by the sync helper."""

    def __init__(self, uses_dynamic_vision: bool, *, nested: bool = True) -> None:
        vision = _VisionStub(uses_dynamic_vision)
        if nested:
            inner = type("_Inner", (), {"visual": vision})()
            self.model = inner
        else:
            self.visual = vision


def _make_processor(prior_dynamic_vision: object = "unset") -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``.

    ``prior_dynamic_vision="unset"`` leaves the class-level default untouched;
    otherwise the given value is written into the instance ``__dict__`` to
    simulate a config-derived or user-explicit assignment.
    """
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    if prior_dynamic_vision != "unset":
        proc.dynamic_vision = prior_dynamic_vision
    return proc


def test_top_level_config_dynamic_vision_default_false() -> None:
    """Preserve backward compatibility: dynamic_vision defaults to False."""
    config = MobilintQwen3VLConfig()
    assert config.dynamic_vision is False


def test_top_level_config_dynamic_vision_round_trip() -> None:
    """Preserve ``dynamic_vision`` through ``to_dict`` / ``from_dict`` round-trip."""
    config = MobilintQwen3VLConfig(dynamic_vision=True)
    assert config.dynamic_vision is True

    payload = config.to_dict()
    assert payload["dynamic_vision"] is True

    restored = MobilintQwen3VLConfig.from_dict(payload)
    assert restored.dynamic_vision is True


def test_vision_config_has_no_dynamic_vision_attribute() -> None:
    """The vision sub-config no longer carries ``dynamic_vision``.

    ``dynamic_vision`` is a release-level attribute that ties the vision
    MXQ, text MXQ, image processor, and video processor together, so it
    lives on the composite config. Regressing the field back onto the
    vision sub-config would nest it as if it were vision-only.
    """
    vision_config = MobilintQwen3VLVisionConfig()
    assert not hasattr(vision_config, "dynamic_vision")


def test_resolve_dynamic_vision_flag_static() -> None:
    """1-input MXQ is the static path."""
    assert MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(1) is False


def test_resolve_dynamic_vision_flag_dynamic() -> None:
    """3-input MXQ ([rope, pos, folded]) is the dynamic path."""
    assert MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(3) is True


def test_resolve_dynamic_vision_flag_rejects_unknown_input_count() -> None:
    """Reject unrecognized MXQ signatures rather than guessing."""
    with pytest.raises(ValueError, match="1 \\(static\\) or 3 \\(dynamic"):
        MobilintQwen3VLVisionModel._resolve_dynamic_vision_flag(2)


def test_reconcile_dynamic_vision_warns_on_config_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn (and trust the MXQ) when top-level config disagrees with detection."""
    config = MobilintQwen3VLConfig(dynamic_vision=False)
    with caplog.at_level(logging.WARNING, logger="mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl"):
        detected = MobilintQwen3VLModel._reconcile_dynamic_vision(config, detected=True)
    assert detected is True
    assert config.dynamic_vision is True
    assert any("disagrees with vision MXQ detection" in rec.message for rec in caplog.records)


def test_reconcile_dynamic_vision_silent_when_matched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning when config and detection agree; config is left aligned."""
    config = MobilintQwen3VLConfig(dynamic_vision=True)
    with caplog.at_level(logging.WARNING, logger="mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl"):
        detected = MobilintQwen3VLModel._reconcile_dynamic_vision(config, detected=True)
    assert detected is True
    assert config.dynamic_vision is True
    assert not any("disagrees with vision MXQ detection" in rec.message for rec in caplog.records)


def test_reconcile_dynamic_vision_reads_visual_when_detected_omitted() -> None:
    """The helper can pull the detected value straight off a vision submodule."""
    config = MobilintQwen3VLConfig(dynamic_vision=False)
    detected = MobilintQwen3VLModel._reconcile_dynamic_vision(config, visual=_VisionStub(uses_dynamic_vision=True))
    assert detected is True
    assert config.dynamic_vision is True


def test_sync_dynamic_vision_from_model_upgrades_default_to_true() -> None:
    """Class default (False) upgrades to detected True with no prior assignment."""
    proc = _make_processor()
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True))
    assert proc.dynamic_vision is True
    assert proc.video_processor.dynamic_vision is True


def test_sync_dynamic_vision_from_model_overwrites_config_derived_value() -> None:
    """A config-derived ``dynamic_vision=False`` is silently replaced by detected True.

    This is the regression the fix addresses: ``from_pretrained`` always writes a
    config-derived value into ``__dict__``, so any lingering mismatch guard
    would raise here even though the caller invoked the helper precisely to
    resolve that mismatch.
    """
    proc = _make_processor(prior_dynamic_vision=False)
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True))
    assert proc.dynamic_vision is True
    assert proc.video_processor.dynamic_vision is True


def test_sync_dynamic_vision_from_model_overwrites_user_override() -> None:
    """An explicit user override also loses to the detected value."""
    proc = _make_processor(prior_dynamic_vision=True)
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=False))
    assert proc.dynamic_vision is False
    assert proc.video_processor.dynamic_vision is False


def test_sync_dynamic_vision_from_model_accepts_flat_model_layout() -> None:
    """Some Qwen3-VL variants expose ``visual`` directly, not under ``.model``."""
    proc = _make_processor()
    proc.sync_dynamic_vision_from_model(_ModelStub(uses_dynamic_vision=True, nested=False))
    assert proc.dynamic_vision is True


def test_sync_dynamic_vision_from_model_rejects_non_qwen3_vl_model() -> None:
    """A model whose vision submodule lacks ``_uses_dynamic_vision`` is rejected."""
    proc = _make_processor()

    class _AlienModel:
        model = type("_Inner", (), {"visual": object()})()

    with pytest.raises(ValueError, match="_uses_dynamic_vision"):
        proc.sync_dynamic_vision_from_model(_AlienModel())


def test_strip_video_outer_wrap_removes_chat_template_wrap() -> None:
    """The chat-template outer ``<|vision_start|><|video_pad|><|vision_end|>`` collapses to ``<|video_pad|>``."""
    text = (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>Describe this video."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    stripped = MobilintQwen3VLProcessor._strip_video_outer_wrap(text)
    assert "<|vision_start|><|video_pad|><|vision_end|>" not in stripped
    assert stripped.count("<|video_pad|>") == 1
    assert "Describe this video." in stripped


def test_strip_video_outer_wrap_leaves_image_wrap_untouched() -> None:
    """Image ``<|vision_start|><|image_pad|><|vision_end|>`` must stay intact.

    Upstream ``replace_image_token`` returns plain ``<|image_pad|>*N`` without
    per-image vision markers, so the outer wrap is the only boundary marker
    for the image's visual region.
    """
    text = "<|vision_start|><|image_pad|><|vision_end|>What is this?"
    stripped = MobilintQwen3VLProcessor._strip_video_outer_wrap(text)
    assert stripped == text


def test_strip_video_outer_wrap_handles_multiple_videos() -> None:
    """Multiple video wraps in a single message all collapse."""
    text = "<|vision_start|><|video_pad|><|vision_end|> and <|vision_start|><|video_pad|><|vision_end|>"
    stripped = MobilintQwen3VLProcessor._strip_video_outer_wrap(text)
    assert stripped == "<|video_pad|> and <|video_pad|>"


def test_strip_video_outer_wrap_batched_list() -> None:
    """List-of-strings batch input is normalized per-item."""
    batch = [
        "<|vision_start|><|video_pad|><|vision_end|>A",
        "no wrap here",
    ]
    stripped = MobilintQwen3VLProcessor._strip_video_outer_wrap(batch)
    assert stripped == ["<|video_pad|>A", "no wrap here"]


def test_strip_video_outer_wrap_noop_on_bare_video_pad() -> None:
    """Bare ``<|video_pad|>`` (already normalized / vLLM-style input) is left alone."""
    text = "prefix <|video_pad|> suffix"
    stripped = MobilintQwen3VLProcessor._strip_video_outer_wrap(text)
    assert stripped == text


# ---------------------------------------------------------------------------
# ``_clamp_dynamic_image_size`` cross-version (tf 4.x dict / tf 5.x SizeDict)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SizeDictLike:
    """Minimal stand-in for tf 5.x's frozen ``SizeDict`` dataclass.

    Real ``SizeDict`` is a ``@dataclass(frozen=True)`` that exposes
    ``__getitem__`` but is *not* a ``Mapping`` (no ``keys()``), so
    ``**size_obj`` unpacking raises ``TypeError``. This stub replicates
    that surface with just the two fields the clamp touches.
    """

    longest_edge: Optional[int] = None
    shortest_edge: Optional[int] = None

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


class _ImageProcessorStub:
    def __init__(self, size, patch_size: int = 14) -> None:
        self.size = size
        self.patch_size = patch_size


def _make_processor_with_image_processor(size) -> MobilintQwen3VLProcessor:
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = _ImageProcessorStub(size=size, patch_size=14)
    proc.max_vision_tokens = _NPU_MAX_VISION_TOKENS
    return proc


def test_clamp_dynamic_image_size_plain_dict_caps_longest_edge() -> None:
    """tf 4.x path: ``size`` is a plain dict and gets replaced in-place."""
    limit = _NPU_MAX_VISION_TOKENS * 14**2
    proc = _make_processor_with_image_processor({"longest_edge": limit * 4, "shortest_edge": limit * 2})
    proc._clamp_dynamic_image_size()
    assert isinstance(proc.image_processor.size, dict)
    assert proc.image_processor.size["longest_edge"] == limit
    assert proc.image_processor.size["shortest_edge"] == limit


def test_clamp_dynamic_image_size_plain_dict_preserves_small_shortest_edge() -> None:
    """``shortest_edge`` below the limit must not be inflated."""
    limit = _NPU_MAX_VISION_TOKENS * 14**2
    proc = _make_processor_with_image_processor({"longest_edge": limit * 4, "shortest_edge": 3136})
    proc._clamp_dynamic_image_size()
    assert proc.image_processor.size["shortest_edge"] == 3136


def test_clamp_dynamic_image_size_size_dict_caps_longest_edge() -> None:
    """tf 5.x path: ``size`` is a frozen ``SizeDict``; clamp must not raise
    ``TypeError`` from ``**size`` unpacking and must yield a new instance."""
    limit = _NPU_MAX_VISION_TOKENS * 14**2
    original = _SizeDictLike(longest_edge=limit * 4, shortest_edge=limit * 2)
    proc = _make_processor_with_image_processor(original)
    proc._clamp_dynamic_image_size()
    new_size = proc.image_processor.size
    assert isinstance(new_size, _SizeDictLike)
    assert new_size is not original  # dataclasses.replace returns a fresh instance
    assert new_size.longest_edge == limit
    assert new_size.shortest_edge == limit


def test_clamp_dynamic_image_size_noop_when_already_within_limit() -> None:
    """Neither the dict nor the SizeDict path mutates when already under budget."""
    limit = _NPU_MAX_VISION_TOKENS * 14**2
    small_dict = {"longest_edge": limit // 2, "shortest_edge": limit // 4}
    proc_dict = _make_processor_with_image_processor(small_dict)
    proc_dict._clamp_dynamic_image_size()
    assert proc_dict.image_processor.size is small_dict

    small_sd = _SizeDictLike(longest_edge=limit // 2, shortest_edge=limit // 4)
    proc_sd = _make_processor_with_image_processor(small_sd)
    proc_sd._clamp_dynamic_image_size()
    assert proc_sd.image_processor.size is small_sd


# ---------------------------------------------------------------------------
# ``from_pretrained`` kwarg propagation to the follow-up config load.
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Stand-in for a top-level Qwen3-VL config with an explicit dynamic_vision."""

    def __init__(self, dynamic_vision: bool) -> None:
        self.dynamic_vision = dynamic_vision


class _LegacyConfigNoDynamicVision:
    """Older static release: config loads, but the ``dynamic_vision`` field is absent."""


def _build_bare_processor() -> MobilintQwen3VLProcessor:
    """Return a ``MobilintQwen3VLProcessor`` instance without running its heavy init.

    ``from_pretrained`` needs the super() return value to satisfy
    ``isinstance(processor, cls)`` and to carry a ``video_processor``
    attribute for ``_sync_dynamic_vision_to_video_processor``.
    """
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    return proc


def test_from_pretrained_forwards_subfolder_to_config_load() -> None:
    """A caller-supplied ``subfolder`` must reach the follow-up AutoConfig call.

    This is the concrete Codex-review regression: ``from_pretrained(path,
    subfolder='release')`` used to load the processor from ``path/release``
    but drop ``subfolder`` before reading the config, so the config lookup
    fell back to ``path/config.json`` and ``dynamic_vision`` silently
    defaulted to False — turning a dynamic-vision MXQ release into a hard
    failure on any multi-image/video input.
    """
    fake_config = _FakeConfig(dynamic_vision=True)
    with (
        mock.patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            return_value=_build_bare_processor(),
        ),
        mock.patch.object(
            processing_qwen3_vl.AutoConfig,
            "from_pretrained",
            return_value=fake_config,
        ) as mock_autoconfig,
    ):
        proc = MobilintQwen3VLProcessor.from_pretrained("some/repo", subfolder="release")

    _, config_kwargs = mock_autoconfig.call_args
    assert config_kwargs.get("subfolder") == "release"
    assert proc.dynamic_vision is True
    assert proc.video_processor.dynamic_vision is True


def test_from_pretrained_forwards_full_hf_loading_kwargs() -> None:
    """Every standard HF loading kwarg the caller passes must reach AutoConfig."""
    fake_config = _FakeConfig(dynamic_vision=False)
    caller_kwargs = {
        "cache_dir": "/tmp/hf-cache",
        "force_download": True,
        "resume_download": False,
        "proxies": {"http": "http://proxy.example:3128"},
        "token": "hf_TOKEN",
        "local_files_only": True,
        "revision": "abc123",
        "subfolder": "release",
        "trust_remote_code": False,
        "code_revision": "def456",
    }
    with (
        mock.patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            return_value=_build_bare_processor(),
        ),
        mock.patch.object(
            processing_qwen3_vl.AutoConfig,
            "from_pretrained",
            return_value=fake_config,
        ) as mock_autoconfig,
    ):
        MobilintQwen3VLProcessor.from_pretrained("some/repo", **caller_kwargs)

    _, config_kwargs = mock_autoconfig.call_args
    for key, value in caller_kwargs.items():
        assert config_kwargs.get(key) == value, f"{key} not forwarded"


def test_from_pretrained_respects_local_files_only() -> None:
    """``local_files_only=True`` must reach AutoConfig (proving no network).

    We assert on the propagated kwarg rather than intercepting the network
    layer: propagation is exactly what the fix owns, and any downstream
    behavior around ``local_files_only`` is upstream HF territory.
    """
    fake_config = _FakeConfig(dynamic_vision=True)
    with (
        mock.patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            return_value=_build_bare_processor(),
        ),
        mock.patch.object(
            processing_qwen3_vl.AutoConfig,
            "from_pretrained",
            return_value=fake_config,
        ) as mock_autoconfig,
    ):
        MobilintQwen3VLProcessor.from_pretrained("some/repo", local_files_only=True)

    _, config_kwargs = mock_autoconfig.call_args
    assert config_kwargs.get("local_files_only") is True


def test_from_pretrained_legacy_config_without_dynamic_vision_silent_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legacy config that omits the field resolves to False *silently*.

    The exception-path fallback logs a warning to surface silent
    misconfiguration, but the load-succeeded-without-field path is the
    legitimate legacy release shape and must not emit a warning.
    """
    with (
        mock.patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            return_value=_build_bare_processor(),
        ),
        mock.patch.object(
            processing_qwen3_vl.AutoConfig,
            "from_pretrained",
            return_value=_LegacyConfigNoDynamicVision(),
        ),
    ):
        with caplog.at_level(
            logging.WARNING,
            logger="mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl",
        ):
            proc = MobilintQwen3VLProcessor.from_pretrained("some/repo")

    assert proc.dynamic_vision is False
    assert proc.video_processor.dynamic_vision is False
    assert not any("dynamic_vision" in rec.message for rec in caplog.records), (
        "Legacy config path must not emit a warning."
    )


def test_from_pretrained_warns_when_config_load_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Config-load failure falls back to False *with* a visible warning.

    A transient error (network, permissions) or a missing ``config.json``
    both land here. The prior implementation swallowed the exception at
    ``debug`` level, hiding the misconfiguration; the fix logs at
    ``warning`` so operators can spot it before the processor silently
    hard-fails dynamic-only inputs.
    """
    with (
        mock.patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            return_value=_build_bare_processor(),
        ),
        mock.patch.object(
            processing_qwen3_vl.AutoConfig,
            "from_pretrained",
            side_effect=OSError("simulated network failure"),
        ),
    ):
        with caplog.at_level(
            logging.WARNING,
            logger="mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl",
        ):
            proc = MobilintQwen3VLProcessor.from_pretrained("some/repo")

    assert proc.dynamic_vision is False
    assert any("simulated network failure" in rec.message for rec in caplog.records)
