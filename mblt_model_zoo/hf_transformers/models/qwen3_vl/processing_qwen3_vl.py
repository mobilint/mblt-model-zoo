import re
from dataclasses import replace as _dataclass_replace
from typing import Callable, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import INTER_CUBIC
from cv2 import resize as cv2_resize
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, load_image
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.video_processing_auto import AutoVideoProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.generic import logging
from transformers.video_utils import VideoInput

from .configuration_qwen3_vl import MobilintQwen3VLConfig

logger = logging.get_logger(__name__)

# NPU vision model fixed input shape: (H_npu, W_npu, C_npu) = (1024, 64, 6)
_NPU_H, _NPU_W = 1024, 64

# transformers 5.x's Qwen3VLProcessor.replace_video_token expands `<|video_pad|>`
# to a per-frame string that already carries its own `<|vision_start|>...<|vision_end|>`
# pairs, so the chat template's outer `<|vision_start|><|video_pad|><|vision_end|>`
# wrap becomes doubly nested. tf 4.x's processor stripped that outer pair itself;
# tf 5.x does not. We pre-normalize the text before dispatch so both versions
# emit the same single-wrap prompt.
_VIDEO_OUTER_WRAP_RE = re.compile(
    r"<\|vision_start\|>\s*<\|video_pad\|>\s*<\|vision_end\|>"
)

# The dynamic vision MXQ takes the pre-merge patch sequence as `inputs[0]`. Its op
# descriptor declares a 4096-token ceiling, but anything above 2048 hangs the NPU
# (watchdog timeout -> `Model_NotAlive`) rather than erroring out, so the default is
# the largest length measured to run. Override `max_vision_tokens` for an MXQ that
# supports longer sequences.
#
# Enforcement strategy — cap-preprocessing (Option 1 from the design). The
# stored processor defaults are capped once at load time by
# ``_clamp_dynamic_image_size`` / ``_clamp_dynamic_video_size`` so the default
# path is safe; caller overrides (``size``, ``max_pixels``, ``min_pixels``,
# ``do_resize``, in either the top-level kwargs or the nested
# ``images_kwargs`` / ``videos_kwargs``) are re-clamped at call time by
# ``_clamp_dynamic_image_call_kwargs`` / ``_clamp_dynamic_video_call_kwargs``
# right before dispatch to ``super().__call__``. ``do_resize=False`` is hard-
# rejected because it strips the ceiling entirely — the resulting patch count
# would depend on raw pixel resolution with no upper bound.
_NPU_MAX_VISION_TOKENS = 2048

# Structural vision kwargs baked into the vision MXQ at compile time. The
# folded feature width handed to the language model at the vision-language
# boundary is ``patch_size * merge_size``, and the temporal stride is
# ``temporal_patch_size``; the MXQ was compiled for a single choice of each.
# ``Qwen3VLImagesKwargs`` and ``Qwen3VLVideosKwargs`` still let a caller pass
# these fields at call time, but any deviation from the shipped processor
# value breaks the boundary shape (silent shape mismatch downstream) *and*
# bypasses the token-budget guard, which derives its ceiling from the stored
# ``patch_size``. Reject caller overrides at the processor surface before
# either failure can materialize.
_STRUCTURAL_VISION_KWARGS = ("patch_size", "temporal_patch_size", "merge_size")

# Resize-shaping caller kwargs that must not touch the static-vision path.
# The static release ships a vision MXQ compiled for a rigid
# ``(H, W, C) = (_NPU_H, _NPU_W, 6)`` grid; ``__call__`` pre-resizes every
# input to the matching pixel resolution via ``_resize_images``, but the
# upstream image processor still runs after that and honors caller-supplied
# ``size`` / ``min_pixels`` / ``max_pixels`` overrides, which would re-resize
# the just-normalized image and break the fixed grid the MXQ expects — either
# shape-mismatching at dispatch or silently producing semantically wrong
# output. ``do_resize`` is checked separately because only ``do_resize=False``
# is a semantic change worth rejecting (``True`` matches the default and the
# static branch's assumption).
_STATIC_RESIZE_OVERRIDE_KWARGS = ("size", "min_pixels", "max_pixels")

# Standard Hugging Face loading kwargs that select which artifact
# ``from_pretrained`` reads. The processor and the model config must resolve
# to the *same* release, so any caller-supplied value here must be propagated
# from the processor's ``from_pretrained`` to the follow-up
# ``AutoConfig.from_pretrained``. Missing any one of these (``subfolder`` in
# particular) makes the two calls disagree — the processor lands in a
# subfolder while the config lookup falls back to the repo root and silently
# picks up the wrong ``dynamic_vision`` (or none at all).
_HF_LOADING_KWARGS = (
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "token",
    "local_files_only",
    "revision",
    "subfolder",
    "trust_remote_code",
    "code_revision",
)


_IMAGE_PAD_TOKEN = "<|image_pad|>"


# Sizes that qualify as a channel dimension when detecting layout in
# ``_to_bhwc``. Grayscale (1), RGB (3), and RGBA (4) cover the layouts
# ``ImageInput`` actually carries; any other size on an endpoint axis is
# treated as spatial. When both candidate axes qualify (e.g. a small
# ``(3, 3, 3)`` array or a ``(N, 3, 3, 3)`` batch), tie-break to HWC / BHWC to
# match the majority upstream convention.
_CHANNEL_CANDIDATES = (1, 3, 4)


def _to_bhwc(img):
    """Normalize a rank-3 or rank-4 image to batch-first channels-last layout.

    Returns ``(bhwc, restore_layout_fn)``. ``bhwc`` is a rank-4
    ``(N, H, W, C)`` array or tensor of the same underlying type as ``img``;
    ``restore_layout_fn`` reverses the transform on a resized ``(N, H', W', C)``
    result so callers get their original layout back. Together they cover the
    eight ``(rank-3 vs rank-4) x (HWC vs CHW) x (ndarray vs torch tensor)``
    combinations without per-shape branching at each call site — cv2 always
    sees an HWC frame, and ``F.interpolate`` always starts from BHWC and
    permutes to BCHW just before the call.

    The channel axis is detected from the endpoint sizes: axis 0 (rank-3) or
    axis 1 (rank-4) vs axis -1. An axis qualifies as ``channels`` when its
    size is in :data:`_CHANNEL_CANDIDATES`. Ambiguous shapes where both
    endpoints qualify tie-break to HWC / BHWC.
    """
    is_torch = torch.is_tensor(img)
    if not is_torch and not isinstance(img, np.ndarray):
        raise TypeError(f"_to_bhwc expects ndarray or torch tensor, got {type(img)}")
    rank = img.ndim
    if rank not in (3, 4):
        raise ValueError(f"_to_bhwc expects a rank-3 or rank-4 image, got rank {rank}")
    first_axis = 0 if rank == 3 else 1
    channels_first = img.shape[first_axis] in _CHANNEL_CANDIDATES
    channels_last = img.shape[-1] in _CHANNEL_CANDIDATES
    # Tie-break to HWC / BHWC when both endpoints look like plausible channel
    # counts, matching the majority upstream convention.
    is_channels_first = channels_first and not channels_last

    if is_torch:
        if rank == 3:
            bhwc = img.permute(1, 2, 0).unsqueeze(0) if is_channels_first else img.unsqueeze(0)
        else:
            bhwc = img.permute(0, 2, 3, 1) if is_channels_first else img
    else:
        if rank == 3:
            bhwc = np.transpose(img, (1, 2, 0))[None, ...] if is_channels_first else img[None, ...]
        else:
            bhwc = np.transpose(img, (0, 2, 3, 1)) if is_channels_first else img

    def restore(bhwc_out):
        if is_torch:
            if rank == 3:
                out = bhwc_out.squeeze(0)
                return out.permute(2, 0, 1) if is_channels_first else out
            return bhwc_out.permute(0, 3, 1, 2) if is_channels_first else bhwc_out
        if rank == 3:
            out = bhwc_out[0]
            return np.transpose(out, (2, 0, 1)) if is_channels_first else out
        return np.transpose(bhwc_out, (0, 3, 1, 2)) if is_channels_first else bhwc_out

    return bhwc, cast(Callable, restore)


def _count_chat_message_images(messages) -> int:
    """Count image content items in a single chat-message conversation.

    A conversation is a list of ``{"role": ..., "content": ...}`` dicts.
    ``content`` is either a plain string (no images) or a list of content
    parts; an image part is a dict with ``type == "image"`` (or the
    ``image``/``image_url``/``image_path`` legacy aliases some Qwen3-VL
    examples still emit). Each such part maps 1:1 to an ``<|image_pad|>``
    placeholder in the rendered chat template output, so counting them
    structurally mirrors placeholder counting on the rendered string
    without having to call ``apply_chat_template`` (which would have side
    effects and requires a bound tokenizer).
    """
    image_types = {"image", "image_url", "image_path"}
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            for part in content:
                if isinstance(part, dict) and part.get("type") in image_types:
                    total += 1
    return total


def _per_prompt_image_counts(text, image_token: str = _IMAGE_PAD_TOKEN) -> list[int]:
    """Return the per-prompt image count for each prompt described by ``text``.

    This is the single source of truth for how the processor associates
    images with prompts. It mirrors the association order in upstream
    ``Qwen3VLProcessor.__call__``: ``text`` is normalized to a list of
    prompts (a bare string becomes a single-element list), each prompt is
    then walked left-to-right, and every ``<|image_pad|>`` placeholder
    consumes one image from the *flat* image input. Container nesting on
    the ``images`` side is irrelevant — only the placeholder counts in
    each prompt determine the per-prompt image count. Guarding on those
    counts avoids both the false-positive (flat images with one
    placeholder per prompt) and the false-negative (nested images with
    both placeholders in the same prompt) that a container-shape
    heuristic produces.

    Supported ``text`` shapes:

    * ``str`` — one rendered prompt.
    * ``list[str]`` — batch of rendered prompts.
    * ``PreTokenizedInput`` (a single list of pre-tokenized token strings)
      — treated as one prompt; counts full-token equality against
      ``image_token``.
    * ``list[PreTokenizedInput]`` — batch of pre-tokenized prompts; one
      count per inner list.
    * A single chat-message conversation (``list[dict]``) — treated as
      one prompt; counts image-typed content parts, which is exactly what
      ``apply_chat_template`` would emit as ``<|image_pad|>`` on the
      rendered string.
    * A batch of chat-message conversations (``list[list[dict]]``) — one
      count per conversation.

    Pre-tokenized shapes are nominal — upstream's ``__call__`` does not
    actually run ``.replace`` on pre-tokenized token lists — but
    supporting them here keeps the guard type-consistent with the
    ``__call__`` signature.
    """
    if text is None:
        return []
    if isinstance(text, str):
        return [text.count(image_token)]
    if not isinstance(text, (list, tuple)):
        return [0]
    # Single chat-message conversation: a list of role/content dicts.
    if text and isinstance(text[0], dict):
        return [_count_chat_message_images(text)]
    counts: list[int] = []
    for entry in text:
        if isinstance(entry, str):
            counts.append(entry.count(image_token))
        elif isinstance(entry, (list, tuple)):
            # Nested list: either a batched chat-message conversation
            # (list of role/content dicts) or a pre-tokenized token list.
            if entry and isinstance(entry[0], dict):
                counts.append(_count_chat_message_images(entry))
            else:
                counts.append(sum(1 for tok in entry if tok == image_token))
        else:
            counts.append(0)
    return counts


def _compute_npu_frame_size(patch_size: int, merge_size: int) -> tuple[int, int]:
    """Derive the pixel resolution that produces the NPU-compatible grid."""
    pw = _NPU_W // (merge_size ** 2)
    gh_merged = int((_NPU_H // pw) ** 0.5)
    side = gh_merged * merge_size * patch_size
    return (side, side)


def _update_size(size_obj, **updates):
    """Return an updated size, transparently across transformers versions.

    tf < 5 stores ``image_processor.size`` as a plain ``dict``; tf >= 5 wraps
    it in a frozen ``SizeDict`` dataclass (readable via ``[key]`` but not a
    ``Mapping``, so ``**size_obj`` unpacking raises ``TypeError``). Use
    ``dataclasses.replace`` for the dataclass case to build a new instance.
    """
    if isinstance(size_obj, dict):
        return {**size_obj, **updates}
    return _dataclass_replace(size_obj, **updates)


def _size_get(size_obj, key: str):
    """Read ``key`` from a size that may be a dict or a frozen ``SizeDict``.

    Mirrors :func:`_update_size` on the read side: both size shapes support
    ``[key]``, but the frozen dataclass raises ``KeyError`` (via ``__getitem__``)
    on missing fields while dicts raise the same, so route through the safe
    ``.get`` / ``getattr`` paths and return ``None`` when the field is absent.
    """
    if isinstance(size_obj, dict):
        return size_obj.get(key)
    return getattr(size_obj, key, None)


class MobilintQwen3VLVideoProcessor(Qwen3VLVideoProcessor):
    """Force NPU-compatible frame size before upstream `_preprocess`.

    When `dynamic_vision` is True the NPU accepts variable resolutions, so we
    delegate straight to the upstream implementation. Otherwise every frame is
    bicubic-resized to the fixed grid derived from `_compute_npu_frame_size`.
    """

    dynamic_vision = False

    def _preprocess(self, videos, do_resize=True, size=None, **kwargs):
        if self.dynamic_vision:
            return super()._preprocess(videos, do_resize=do_resize, size=size, **kwargs)
        target = _compute_npu_frame_size(self.patch_size, self.merge_size)
        resized = [
            F.interpolate(v.float(), size=target, mode="bicubic", align_corners=False)
            for v in videos
        ]
        return super()._preprocess(resized, do_resize=False, size=size, **kwargs)


class MobilintQwen3VLProcessor(Qwen3VLProcessor):
    dynamic_vision = False
    max_vision_tokens = _NPU_MAX_VISION_TOKENS

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        # AutoVideoProcessor loads the vanilla Qwen3VLVideoProcessor from the
        # HF config; rebuild it as our subclass so `_preprocess` is our override.
        if video_processor is not None and not isinstance(
            video_processor, MobilintQwen3VLVideoProcessor
        ):
            video_processor = MobilintQwen3VLVideoProcessor(**video_processor.to_dict())
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def _sync_dynamic_vision_to_video_processor(self) -> None:
        vp = getattr(self, "video_processor", None)
        if isinstance(vp, MobilintQwen3VLVideoProcessor):
            vp.dynamic_vision = bool(self.dynamic_vision)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if not isinstance(processor, cls):
            # AutoProcessor resolved a different class; leave it untouched.
            return processor
        # Forward every standard HF loading kwarg the caller supplied so the
        # processor and the config are read from the *same* artifact. In
        # particular, ``subfolder`` must reach ``AutoConfig`` — otherwise a
        # release that lives in ``<repo>/release/`` loads the processor from
        # the subfolder but the config lookup falls back to ``<repo>/``, and
        # ``dynamic_vision`` is silently defaulted to ``False``.
        config_kwargs = {k: kwargs[k] for k in _HF_LOADING_KWARGS if k in kwargs}
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        except Exception as exc:
            # A legacy static release that ships a config *without* the
            # ``dynamic_vision`` field is handled by the ``getattr`` default
            # in the success path below; it does not reach this branch. So
            # any exception here means the config load itself failed
            # (missing ``config.json``, transient network/permission error,
            # etc.), which would silently drop the processor into static
            # mode. Warn loudly so the misconfiguration is easy to spot.
            logger.warning(
                "Could not load model config from %r to determine "
                "dynamic_vision; defaulting to False. This is expected only "
                "for a processor artifact with no accompanying config.json; "
                "for a transient network/permission failure the processor "
                "will hard-fail dynamic-only inputs even against a "
                "dynamic-vision MXQ. Underlying error: %s",
                pretrained_model_name_or_path,
                exc,
            )
            vision_dyn = False
        else:
            vision_dyn = bool(getattr(config, "dynamic_vision", False))
        processor.dynamic_vision = vision_dyn
        processor._sync_dynamic_vision_to_video_processor()
        return processor

    def sync_dynamic_vision_from_model(self, model) -> None:
        """Adopt ``dynamic_vision`` from a loaded Qwen3-VL model's paired MXQs.

        Vision and text MXQs are a *bundled release* — one cannot be swapped
        independently, because a dynamic-vision MXQ produces per-image RoPE
        tensors that the text MXQ must consume via its rope input. This
        helper consults both compiled signatures and refuses to enable
        dynamic mode unless they agree.

        In the typical flow, ``dynamic_vision`` is populated automatically by
        :meth:`from_pretrained` reading the top-level ``config.dynamic_vision``
        from the shipped ``config.json``, so calling this helper is
        unnecessary.

        Use it only when the processor and model have diverged at runtime —
        e.g. the model was loaded with an explicit ``vision_mxq_path=`` kwarg
        pointing at an MXQ whose signature (static/dynamic) doesn't match the
        config the processor was built from.

        Each submodule detects its own signature from its compiled MXQ:

        * ``visual._uses_dynamic_vision`` is True for a 3-input vision MXQ.
        * ``language_model._uses_rope_input`` is True for a 3-input text MXQ
          that receives a per-image rope tensor.

        When the flags agree, this helper overwrites ``dynamic_vision`` (and
        the video processor's mirror) with the agreed value. Any prior value
        — the class default, a config-derived assignment from
        :meth:`from_pretrained`, or an explicit user override — is replaced.
        When they disagree, it raises ``ValueError`` naming both flags and
        both MXQ paths so the caller can either load a consistent release or
        override both ``vision_mxq_path=`` and ``text_mxq_path=`` to a
        matching pair.
        """
        submodule_root = getattr(model, "model", model)
        vision = getattr(submodule_root, "visual", None)
        if vision is None or not hasattr(vision, "_uses_dynamic_vision"):
            raise ValueError(
                "sync_dynamic_vision_from_model expects a Qwen3-VL model whose "
                "vision submodule exposes `_uses_dynamic_vision`."
            )
        language_model = getattr(submodule_root, "language_model", None)
        if language_model is None or not hasattr(language_model, "_uses_rope_input"):
            raise ValueError(
                "sync_dynamic_vision_from_model expects a Qwen3-VL model whose "
                "language_model submodule exposes `_uses_rope_input`."
            )
        vision_dynamic = bool(vision._uses_dynamic_vision)
        text_dynamic = bool(language_model._uses_rope_input)
        if vision_dynamic != text_dynamic:
            config = getattr(model, "config", None)
            vision_path = getattr(config, "vision_mxq_path", "<unknown>")
            text_path = getattr(config, "text_mxq_path", "<unknown>")
            raise ValueError(
                "Qwen3-VL vision and text MXQs are a bundled release and cannot "
                "be swapped independently: visual._uses_dynamic_vision="
                f"{vision_dynamic} disagrees with language_model._uses_rope_input="
                f"{text_dynamic}. A dynamic-vision MXQ produces per-image RoPE "
                "tensors that the text MXQ must consume via its rope input; "
                "pairing a 3-input vision MXQ with a legacy 2-input text MXQ "
                "(or vice versa) silently corrupts image-boundary information. "
                f"vision_mxq_path={vision_path!r}, text_mxq_path={text_path!r}. "
                "Load a consistent Qwen3-VL release, or override both "
                "vision_mxq_path= and text_mxq_path= to a matching pair."
            )
        self.dynamic_vision = vision_dynamic
        self._sync_dynamic_vision_to_video_processor()

    @staticmethod
    def _resize_one(img, size=(224, 224)):
        if isinstance(img, str):
            img = load_image(img)
        if isinstance(img, Image.Image):
            return img.resize(size)
        # ``ImageInput`` accepts arrays / tensors in any of the eight layouts
        # from (rank-3 vs rank-4) x (HWC vs CHW) x (ndarray vs torch tensor).
        # cv2.resize only takes an HWC frame and F.interpolate only takes
        # NCHW, so route both types through ``_to_bhwc`` first and let the
        # returned restore closure put the original layout back on the way out.
        if isinstance(img, np.ndarray):
            bhwc, restore = _to_bhwc(img)
            resized = np.stack(
                [
                    cast(
                        np.ndarray,
                        cv2_resize(frame, size[::-1], interpolation=INTER_CUBIC),
                    )
                    for frame in bhwc
                ]
            )
            return restore(resized)
        if torch.is_tensor(img):
            bhwc, restore = _to_bhwc(img)
            bchw = bhwc.permute(0, 3, 1, 2).float()
            resized_bhwc = F.interpolate(
                bchw, size=size, mode="bicubic", align_corners=False
            ).permute(0, 2, 3, 1)
            return restore(resized_bhwc)
        raise TypeError(f"Unsupported image type: {type(img)}")

    @classmethod
    def _resize_images(cls, images):
        if isinstance(images, list):
            return [cls._resize_images(item) for item in images]
        return cls._resize_one(images)

    def _clamp_dynamic_image_size(self) -> None:
        """Cap `max_pixels` so dynamic-vision grids fit the NPU sequence limit.

        The dynamic vision MXQ receives the pre-merge patch sequence as
        `inputs[0]`, so its length is `grid_t * grid_h * grid_w` and must stay
        within `max_vision_tokens`. `smart_resize` guarantees
        `height * width <= max_pixels`, so bounding `max_pixels` by
        `max_vision_tokens * patch_size ** 2` bounds the patch count as well,
        while preserving the aspect ratio and the `patch_size * merge_size`
        grid alignment.

        Capping `max_pixels` rather than pre-resizing via `_resize_images` is
        deliberate: `smart_resize` runs *after* that hook and re-rounds every
        side to a `patch_size * merge_size` multiple, so a pre-resized side can
        be rounded back up and silently overshoot the budget.

        Two surfaces feed ``smart_resize`` and both must be capped. tf 4.x's
        ``Qwen2VLImageProcessor`` (the base of Qwen3-VL's image processor)
        stores ``max_pixels`` / ``min_pixels`` as separate scalar attributes,
        and its ``preprocess()`` falls back to those attributes when the
        caller omits ``max_pixels`` — the effective ``size`` is then derived
        via a "backward-compatibility" branch as
        ``{shortest_edge: self.min_pixels, longest_edge: self.max_pixels}``,
        and ``self.size`` is never even consulted. Updating only ``self.size``
        therefore does nothing on the default no-override path. tf 5.x's
        image processor dropped those scalar attributes, so we guard with a
        ``getattr`` sentinel and skip the scalar branch when they are absent.
        """
        ip = self.image_processor
        limit = self.max_vision_tokens * ip.patch_size ** 2
        current_longest = ip.size["longest_edge"]
        _MISSING = object()
        current_max_pixels = getattr(ip, "max_pixels", _MISSING)
        size_over_budget = current_longest > limit
        scalar_over_budget = (
            current_max_pixels is not _MISSING
            and current_max_pixels is not None
            and current_max_pixels > limit
        )
        if not (size_over_budget or scalar_over_budget):
            return

        logger.info(
            "[dynamic-vision] capped max_pixels %d -> %d (<= %d vision tokens)",
            current_max_pixels if scalar_over_budget else current_longest,
            limit,
            self.max_vision_tokens,
        )
        if size_over_budget:
            ip.size = _update_size(
                ip.size,
                longest_edge=limit,
                shortest_edge=min(ip.size["shortest_edge"], limit),
            )
        if scalar_over_budget:
            ip.max_pixels = limit
            current_min_pixels = getattr(ip, "min_pixels", _MISSING)
            if (
                current_min_pixels is not _MISSING
                and current_min_pixels is not None
                and current_min_pixels > limit
            ):
                ip.min_pixels = limit

    def _clamp_dynamic_image_call_kwargs(self, kwargs: dict) -> None:
        """Cap image-side caller overrides so nothing exceeds the NPU vision-token budget.

        ``_clamp_dynamic_image_size`` caps the stored processor defaults, but the
        caller can still smuggle a bypass through top-level ``kwargs`` (``size``,
        ``max_pixels``, ``min_pixels``, ``do_resize``) or nested
        ``images_kwargs``. Upstream ``_merge_kwargs`` reads both routes: a flat
        top-level kwarg is copied into every modality's kwarg dict (``.get``,
        not ``.pop``), and the nested per-modality dict wins on collision. Any
        one of these can silently produce a grid with
        ``grid_h * grid_w > max_vision_tokens``, which hangs the NPU rather
        than erroring cleanly. Re-clamp both scopes here so the effective values
        stay inside the budget after caller overrides land.

        ``do_resize=False`` is a hard reject: the caller has asked the image
        processor to skip resize entirely, so patch extraction runs at raw
        resolution and the budget has no upper bound. Fail loudly with a
        message pointing at the ceiling rather than silently overriding the
        caller's intent.
        """
        ip = self.image_processor
        limit = self.max_vision_tokens * ip.patch_size ** 2
        for scope in self._call_kwargs_scopes(kwargs, "images_kwargs"):
            self._reject_do_resize_false(scope, "image")
            self._cap_pixel_kwargs(scope, limit, "image")
            self._cap_size_edges(scope, limit, "image")

    def _clamp_dynamic_video_size(self) -> None:
        """Cap `max_pixels` so dynamic-vision video frames fit the NPU sequence limit.

        Mirrors ``_clamp_dynamic_image_size`` for the video processor. The
        dynamic vision MXQ takes the pre-merge patch sequence as ``inputs[0]``
        and hangs the NPU (watchdog timeout -> ``Model_NotAlive``) above
        ``max_vision_tokens`` per frame, so a high-resolution video frame must
        not produce a grid with ``grid_h * grid_w > max_vision_tokens``.

        The video ``smart_resize`` bounds the *volume* ``t_bar * h_bar * w_bar``
        by ``max_pixels`` with ``t_bar >= temporal_patch_size``. That means
        ``h_bar * w_bar <= max_pixels / temporal_patch_size``, so setting
        ``max_pixels = max_vision_tokens * patch_size ** 2 * temporal_patch_size``
        guarantees per-frame ``grid_h * grid_w <= max_vision_tokens`` while
        preserving the aspect ratio and grid alignment.

        Scalar ``max_pixels`` / ``min_pixels`` attributes are handled the same
        way as on the image path: on tf versions that keep them as separate
        fallbacks (``preprocess`` reads them when the caller omits
        ``max_pixels``), we cap them alongside ``size``. On versions where
        the attribute is absent (default for the tf 4.x
        ``Qwen3VLVideoProcessor`` at the time of writing), the ``getattr``
        sentinel skips the scalar branch so the clamp is version-tolerant.
        """
        vp = self.video_processor
        if vp is None:
            return
        limit = self.max_vision_tokens * vp.patch_size ** 2 * vp.temporal_patch_size
        current_longest = vp.size["longest_edge"]
        _MISSING = object()
        current_max_pixels = getattr(vp, "max_pixels", _MISSING)
        size_over_budget = current_longest > limit
        scalar_over_budget = (
            current_max_pixels is not _MISSING
            and current_max_pixels is not None
            and current_max_pixels > limit
        )
        if not (size_over_budget or scalar_over_budget):
            return

        logger.info(
            "[dynamic-vision] capped video max_pixels %d -> %d (<= %d vision tokens/frame)",
            current_max_pixels if scalar_over_budget else current_longest,
            limit,
            self.max_vision_tokens,
        )
        if size_over_budget:
            vp.size = _update_size(
                vp.size,
                longest_edge=limit,
                shortest_edge=min(vp.size["shortest_edge"], limit),
            )
        if scalar_over_budget:
            vp.max_pixels = limit
            current_min_pixels = getattr(vp, "min_pixels", _MISSING)
            if (
                current_min_pixels is not _MISSING
                and current_min_pixels is not None
                and current_min_pixels > limit
            ):
                vp.min_pixels = limit

    def _clamp_dynamic_video_call_kwargs(self, kwargs: dict) -> None:
        """Cap video-side caller overrides so nothing exceeds the NPU vision-token budget.

        Companion to :meth:`_clamp_dynamic_image_call_kwargs` for the video path.
        The video processor doesn't accept ``max_pixels`` / ``min_pixels`` as
        call kwargs (they aren't declared on ``VideosKwargs``), so the attack
        surface here is limited to ``size`` and ``do_resize``. Both are still
        reachable via top-level ``kwargs`` (a flat ``size=`` is copied into
        every modality by ``_merge_kwargs``) or via ``videos_kwargs``.
        """
        vp = self.video_processor
        if vp is None:
            return
        limit = self.max_vision_tokens * vp.patch_size ** 2 * vp.temporal_patch_size
        for scope in self._call_kwargs_scopes(kwargs, "videos_kwargs"):
            self._reject_do_resize_false(scope, "video")
            self._cap_size_edges(scope, limit, "video")

    @staticmethod
    def _call_kwargs_scopes(kwargs: dict, nested_key: str) -> list:
        """Return the top-level kwargs plus the nested per-modality dict when present.

        ``_merge_kwargs`` reads both scopes to build the effective per-modality
        kwarg dict, so both must be re-clamped in place. Non-dict nested
        values (``None``, missing) are dropped rather than raising: they simply
        contribute no overrides.
        """
        scopes = [kwargs]
        nested = kwargs.get(nested_key)
        if isinstance(nested, dict):
            scopes.append(nested)
        return scopes

    def _reject_structural_vision_overrides(
        self,
        kwargs: dict,
        nested_key: str,
        source_attr: str,
        kind: str,
    ) -> None:
        """Reject caller overrides of vision MXQ compile-time structural knobs.

        ``patch_size`` / ``temporal_patch_size`` / ``merge_size`` are baked
        into the compiled vision MXQ. The folded feature width handed to the
        language model at the vision-language boundary is
        ``patch_size * merge_size``, and the temporal stride is
        ``temporal_patch_size``; changing any of them at call time either
        produces a shape mismatch against the MXQ or silently emits a wrong
        grid that the language model reads as plausible-but-wrong tokens. The
        token-budget clamp is *also* fooled — its ceiling is
        ``max_vision_tokens * patch_size ** 2``, computed from the stored
        ``patch_size``, so a smaller caller-supplied ``patch_size`` inflates
        the real patch count past the NPU watchdog boundary.

        Compare each override in the top-level ``kwargs`` and the nested
        per-modality dict against ``self.<source_attr>`` (the shipped image or
        video processor, whose attribute value came from the config). Raise
        ``ValueError`` on any mismatch; pop matching values so the upstream
        call sees a clean dict. The baseline source is looked up lazily so a
        call that supplied no structural kwargs is a no-op even when the
        processor was constructed without that submodule.
        """
        scopes = self._call_kwargs_scopes(kwargs, nested_key)
        if not any(
            field in scope for scope in scopes for field in _STRUCTURAL_VISION_KWARGS
        ):
            return
        baseline_source = getattr(self, source_attr, None)
        for scope in scopes:
            for field in _STRUCTURAL_VISION_KWARGS:
                if field not in scope:
                    continue
                override = scope[field]
                if override is None:
                    scope.pop(field)
                    continue
                baseline = getattr(baseline_source, field, None)
                if baseline is None or override != baseline:
                    raise ValueError(
                        f"{kind} {field}={override!r} cannot override the vision "
                        f"MXQ's compile-time value ({baseline!r}). {field} is a "
                        "release-level structural parameter baked into the compiled "
                        "Qwen3-VL vision MXQ: the folded feature width the language "
                        "model expects at the vision-language boundary is "
                        "patch_size * merge_size, and the temporal stride is "
                        "temporal_patch_size. Overriding it at call time also "
                        "bypasses the NPU vision-token guard (the ceiling is "
                        "derived from the stored patch_size). Remove the override, "
                        "or load a Qwen3-VL release compiled with the desired value."
                    )
                scope.pop(field)

    def _reject_do_resize_false(self, scope: dict, kind: str) -> None:
        if scope.get("do_resize") is False:
            raise ValueError(
                f"do_resize=False bypasses the {self.max_vision_tokens}-token NPU "
                f"vision-token ceiling for {kind} inputs. Larger inputs hang the "
                "NPU (watchdog timeout -> Model_NotAlive) rather than erroring "
                "cleanly. Remove the do_resize override or pre-resize the input "
                "so its resulting patch grid fits the ceiling."
            )

    def _reject_static_image_resize_overrides(self, kwargs: dict) -> None:
        """Reject caller-supplied resize overrides on the static-vision image path.

        A static-vision Qwen3-VL release ships a vision MXQ compiled for a rigid
        ``(_NPU_H, _NPU_W, 6)`` grid derived from the shipped ``patch_size`` /
        ``merge_size``. ``__call__`` pre-resizes every input to the matching
        pixel resolution via ``_resize_images``, but the upstream image
        processor's ``_preprocess`` still runs after that step and honors
        caller-supplied ``size`` / ``min_pixels`` / ``max_pixels`` overrides —
        which re-resize the just-normalized image and break the fixed grid the
        MXQ expects, either shape-mismatching at dispatch or silently producing
        semantically wrong output. The dynamic path treats these overrides as
        first-class inputs and clamps them; the static path is stricter because
        no variance in spatial size is compatible with the compiled MXQ's fixed
        grid.

        Both the top-level ``kwargs`` and the nested ``images_kwargs`` dict
        are inspected. ``do_resize=True`` matches the branch's assumption
        (upstream's default) and is passed through; ``do_resize=False`` is
        a semantic change and rejected — do not silently coerce it, since an
        override that changes semantics is worth surfacing to the caller.
        """
        for scope in self._call_kwargs_scopes(kwargs, "images_kwargs"):
            for field in _STATIC_RESIZE_OVERRIDE_KWARGS:
                if field not in scope:
                    continue
                value = scope[field]
                if value is None:
                    continue
                self._raise_static_resize_override(field, value)
            if scope.get("do_resize") is False:
                self._raise_static_resize_override("do_resize", False)

    @staticmethod
    def _raise_static_resize_override(field: str, value) -> None:
        raise ValueError(
            f"image {field}={value!r} cannot override the static-vision "
            "Qwen3-VL processor's fixed resize. The static release ships a "
            f"vision MXQ compiled for a rigid ({_NPU_H}, {_NPU_W}, 6) grid, "
            "and the processor pre-resizes every input to the matching pixel "
            "resolution before dispatch. Any additional resize would shape-"
            "mismatch the MXQ or silently emit a grid the decoder reads as "
            "plausible-but-wrong tokens. Remove the override, or load a "
            "dynamic-vision Qwen3-VL release for resize control."
        )

    def _cap_pixel_kwargs(self, scope: dict, limit: int, kind: str) -> None:
        """Cap ``max_pixels`` / ``min_pixels`` scalar overrides against ``limit``.

        Bounding ``max_pixels`` bounds ``smart_resize``'s ``h * w`` product,
        which bounds the patch count. Bounding ``min_pixels`` prevents the
        symmetric scale-*up* path where an oversized floor forces the
        rescaler to inflate a small input past the budget.
        """
        for field in ("max_pixels", "min_pixels"):
            value = scope.get(field)
            if value is None or value <= limit:
                continue
            logger.info(
                "[dynamic-vision] capped call-time %s %s %d -> %d (<= %d vision tokens)",
                kind, field, value, limit, self.max_vision_tokens,
            )
            scope[field] = limit

    def _cap_size_edges(self, scope: dict, limit: int, kind: str) -> None:
        """Cap ``size.longest_edge`` / ``size.shortest_edge`` against ``limit``.

        A ``size`` override without those edge keys (e.g. ``{"height", "width"}``)
        is left untouched: the upstream image and video processors both reject
        such a size shape at call time, so no bypass is possible.
        """
        size = scope.get("size")
        if size is None:
            return
        longest = _size_get(size, "longest_edge")
        shortest = _size_get(size, "shortest_edge")
        updates: dict = {}
        if longest is not None and longest > limit:
            updates["longest_edge"] = limit
        if shortest is not None and shortest > limit:
            updates["shortest_edge"] = limit
        if not updates:
            return
        logger.info(
            "[dynamic-vision] capped call-time %s size longest=%s shortest=%s "
            "-> %s (<= %d vision tokens)",
            kind, longest, shortest, updates, self.max_vision_tokens,
        )
        scope["size"] = _update_size(size, **updates)

    def _apply_safety_envelope(
        self,
        images: Optional[ImageInput],
        videos: Optional[VideoInput],
        kwargs: dict,
    ) -> None:
        """Centralized safety gate for the processor's caller-kwargs surface.

        Every kwargs-surface invariant that the loaded release requires lives
        in this one method so a new class of "caller kwargs bypass safety"
        leak surfaces as a change here rather than as a scattered per-site
        patch. ``__call__`` invokes it exactly once, right after the incoming
        kwargs have been assembled and right before dispatch to
        ``super().__call__``. The invariants enforced (in order):

        1. **Structural knob equality.** ``patch_size`` /
           ``temporal_patch_size`` / ``merge_size`` are baked into the vision
           MXQ at compile time — the folded feature width the language model
           expects at the vision-language boundary is ``patch_size *
           merge_size`` and the temporal stride is ``temporal_patch_size``.
           Any override in the top-level or nested per-modality kwargs must
           match the shipped processor/config value; otherwise the boundary
           shape mismatches (or worse, silently reads plausible-but-wrong
           tokens). Run first because a smaller caller-supplied ``patch_size``
           would otherwise fool the vision-token budget clamp (whose ceiling
           is derived from the stored ``patch_size``).

        2. **Static branch resize rejection.** A static-vision release ships
           a vision MXQ compiled for a rigid ``(_NPU_H, _NPU_W, 6)`` grid;
           ``__call__`` pre-resizes every image to the matching pixel
           resolution via ``_resize_images``. Any caller-supplied ``size`` /
           ``min_pixels`` / ``max_pixels`` / ``do_resize=False`` would still
           reach upstream's ``_preprocess`` after that step and re-resize the
           just-normalized image away from the fixed grid. The video branch
           doesn't need a symmetric reject — it hard-fails at ``__call__``
           before the envelope runs on a static release — but the envelope
           still owns the story: it only clamps video kwargs on the dynamic
           branch and hands the reject to ``__call__`` for the static branch.

        3. **Dynamic vision-token budget.** A dynamic-vision release accepts
           variable resolutions, but the vision MXQ hangs the NPU (watchdog
           timeout → ``Model_NotAlive``) above the pre-merge patch ceiling of
           ``self.max_vision_tokens * patch_size ** 2``. Cap the storage
           defaults (``ip.size`` / ``ip.max_pixels``, ``vp.size`` /
           ``vp.max_pixels``) so an omitted-kwarg call is safe by default,
           then re-clamp any per-call ``size`` / ``max_pixels`` /
           ``min_pixels`` overrides in both the top-level and the nested
           per-modality scopes. ``do_resize=False`` is a hard reject because
           it strips the ceiling entirely.

        4. **MRoPE metadata invariant.** tf 5.x's ``Qwen3VLModel.compute_3d_position_ids``
           and the generate-side ``_prepare_position_ids_for_generation``
           build MRoPE 3-D t/h/w positions only when ``mm_token_type_ids``
           is present in the tokenizer output. Without it, both fall back to
           linear (non-MRoPE) positions and the decoder cannot distinguish
           visual tokens by time/space — degenerate output on video, stale
           position math on multi-image. For a multimodal run on tf 5.x,
           force ``text_kwargs['return_mm_token_type_ids'] = True`` by
           explicit assignment (not ``setdefault``) so a caller-supplied
           ``False`` cannot silently disable MRoPE. Gate on
           ``create_mm_token_type_ids`` — the tf 5.x method absent on tf 4.x,
           where ``generate`` strictly rejects unknown ``model_kwargs`` and
           the field is a no-op the tokenizer would refuse.

        Release-level contract hard-fails (video-on-static, per-prompt
        multi-image-on-static) are *not* caller-kwargs invariants — they
        reject inputs the loaded release cannot serve at all — so they
        happen in ``__call__`` before this envelope runs. Storage-level
        defaults (``_clamp_dynamic_image_size`` / ``_clamp_dynamic_video_size``)
        are invoked here as the belt-and-suspenders companion to the
        per-call clamp: the envelope catches caller overrides, the storage
        clamp catches the omitted-kwarg default path (tf 4.x's
        ``Qwen2VLImageProcessor.preprocess`` reads ``self.max_pixels`` from
        the scalar attribute when the caller omits it).
        """
        if images is not None:
            self._reject_structural_vision_overrides(
                kwargs, "images_kwargs", "image_processor", "image"
            )
        if videos is not None:
            self._reject_structural_vision_overrides(
                kwargs, "videos_kwargs", "video_processor", "video"
            )

        if images is not None:
            if self.dynamic_vision:
                self._clamp_dynamic_image_size()
                self._clamp_dynamic_image_call_kwargs(kwargs)
            else:
                self._reject_static_image_resize_overrides(kwargs)

        if videos is not None and self.dynamic_vision:
            self._clamp_dynamic_video_size()
            self._clamp_dynamic_video_call_kwargs(kwargs)

        if (images is not None or videos is not None) and hasattr(
            self, "create_mm_token_type_ids"
        ):
            text_kwargs = kwargs.get("text_kwargs")
            if text_kwargs is None:
                text_kwargs = {}
                kwargs["text_kwargs"] = text_kwargs
            prior = text_kwargs.get("return_mm_token_type_ids")
            if prior is False:
                logger.debug(
                    "[safety-envelope] overwriting caller-supplied "
                    "text_kwargs.return_mm_token_type_ids=False -> True: tf 5.x "
                    "MRoPE builds 3-D t/h/w positions only when "
                    "mm_token_type_ids is present in the tokenizer output"
                )
            text_kwargs["return_mm_token_type_ids"] = True

    @staticmethod
    def _strip_video_outer_wrap(text):
        """Reduce chat-template ``<|vision_start|><|video_pad|><|vision_end|>`` to ``<|video_pad|>``.

        On transformers 5.x the upstream processor only expands ``<|video_pad|>``
        into a per-frame string (which carries its own ``<|vision_start|>``/
        ``<|vision_end|>`` around each frame) and leaves the chat template's
        outer pair intact, producing a double-nested vision structure that
        breaks ``get_rope_index`` / ``visual_pos_masks`` and yields an
        immediate-EOS response. tf 4.x's processor already did this stripping
        via its explicit ``if <vision_start><video_pad><vision_end> in text``
        branch, so on that version the regex is a no-op (the pattern is
        replaced before the branch check would have matched, and the vLLM
        fallback ``else`` produces the same final expansion at the same
        position). Image tokens are left untouched — ``replace_image_token``
        returns a plain ``<|image_pad|>*N`` string with no inner vision
        markers, so the outer wrap around images is the only boundary marker
        upstream has for the image visual region.
        """
        if isinstance(text, str):
            return _VIDEO_OUTER_WRAP_RE.sub("<|video_pad|>", text)
        if isinstance(text, list):
            return [MobilintQwen3VLProcessor._strip_video_outer_wrap(item) for item in text]
        return text

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        assert text is not None, "text is None!"

        # Release-level contract hard-fails. These reject inputs the loaded
        # release cannot serve at all — a static-vision MXQ has neither
        # per-image RoPE (needed for per-prompt multi-image) nor per-frame
        # RoPE / variable visual-token count (needed for video). They are
        # *not* caller-kwargs invariants, so they live here rather than in
        # the safety envelope: no point clamping/rejecting kwargs for a
        # call we would reject anyway, and the safety envelope's static-
        # branch resize reject would obscure the "load a dynamic-vision
        # release" story with a shape-mismatch complaint.
        if videos is not None and not self.dynamic_vision:
            raise NotImplementedError(
                "Video input requires a dynamic-vision Qwen3-VL release (3-input vision "
                "MXQ with variable visual-token count in the text decoder). The currently "
                "loaded processor is in static mode (dynamic_vision=False). Load a "
                "Qwen3-VL release that ships a dynamic vision MXQ, or pass only image "
                "inputs."
            )
        if images is not None and not self.dynamic_vision:
            # Match upstream's per-prompt association order: count
            # ``<|image_pad|>`` placeholders per prompt (a batch of N
            # single-image prompts must pass through — only per-prompt
            # multi-image is the constraint). Container nesting on
            # ``images`` is irrelevant — see ``_per_prompt_image_counts``.
            image_token = getattr(self, "image_token", _IMAGE_PAD_TOKEN)
            per_prompt_counts = _per_prompt_image_counts(text, image_token)
            offender = next(
                ((i, c) for i, c in enumerate(per_prompt_counts) if c > 1),
                None,
            )
            if offender is not None:
                offending_index, offending_count = offender
                raise NotImplementedError(
                    f"Multi-image input at prompt index {offending_index} "
                    f"({offending_count} images bound to that prompt) requires "
                    "a dynamic-vision Qwen3-VL release (3-input vision MXQ with "
                    "per-image 2D RoPE in the text decoder). The currently "
                    "loaded processor is in static mode (dynamic_vision=False), "
                    "which supports exactly one image per prompt. Load a "
                    "Qwen3-VL release that ships a dynamic vision MXQ, or pass "
                    "a single image per prompt."
                )

        # Single centralized gate for every caller-kwargs invariant. See
        # ``_apply_safety_envelope`` for the full contract.
        self._apply_safety_envelope(images, videos, kwargs)

        if images is not None:
            if self.dynamic_vision:
                logger.debug(
                    "[dynamic-vision] skipping forced resize, keeping original aspect ratio"
                )
            else:
                # Force the fixed static-MXQ grid. The envelope's static
                # resize reject already fired for size / min_pixels /
                # max_pixels / do_resize=False overrides that would otherwise
                # re-resize away from this grid inside upstream ``_preprocess``.
                images = self._resize_images(images)

        if videos is not None:
            # Runtime consistency: mirror ``dynamic_vision`` onto the video
            # processor before it sees the frames. Only reached on a dynamic
            # release (the video hard-fail above catches static).
            self._sync_dynamic_vision_to_video_processor()
            text = self._strip_video_outer_wrap(text)

        return super().__call__(images, text, videos, **kwargs)


AutoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLProcessor)
AutoVideoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLVideoProcessor)
