import re
from dataclasses import replace as _dataclass_replace
from typing import Optional, Union, cast

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


def _count_images(images) -> int:
    """Count images in an ``ImageInput`` without loading/decoding.

    ``make_flat_list_of_images`` requires each leaf to be a PIL image or
    ndarray, so it rejects URL/path strings that the processor accepts as
    valid image inputs (they are resolved later, downstream of this guard).
    We only need the count for the multi-image hard-fail, so recurse through
    list/tuple containers and treat every non-container leaf as one image —
    with the standard ndarray/tensor batch axis expansion.
    """
    if images is None:
        return 0
    if isinstance(images, (list, tuple)):
        return sum(_count_images(item) for item in images)
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return images.shape[0]
    if torch.is_tensor(images) and images.ndim == 4:
        return images.shape[0]
    return 1


def _per_prompt_image_pad_counts(text, image_token: str = _IMAGE_PAD_TOKEN) -> list[int]:
    """Return the ``<|image_pad|>`` count for each prompt described by ``text``.

    Mirrors the association order in upstream ``Qwen3VLProcessor.__call__``:
    ``text`` is normalized to a list of prompts (a bare string becomes a
    single-element list), each prompt is then walked left-to-right, and every
    ``<|image_pad|>`` placeholder consumes one image from the *flat* image
    input. Container nesting on the ``images`` side is irrelevant — only the
    placeholder counts in each prompt determine the per-prompt image count.
    Guarding on those counts avoids both the false-positive (flat images with
    one placeholder per prompt) and the false-negative (nested images with
    both placeholders in the same prompt) that a container-shape heuristic
    produces.

    ``PreTokenizedInput`` (a list of pre-tokenized token strings for a single
    prompt) is treated as one prompt and counts full-token equality against
    ``image_token``. A batch of pre-tokenized inputs is a list of such lists
    and yields one count per inner list. This case is nominal — upstream's
    ``__call__`` does not actually run ``.replace`` on pre-tokenized token
    lists — but supporting it here keeps the guard type-consistent with the
    ``__call__`` signature.
    """
    if text is None:
        return []
    if isinstance(text, str):
        return [text.count(image_token)]
    if not isinstance(text, (list, tuple)):
        return [0]
    counts: list[int] = []
    for entry in text:
        if isinstance(entry, str):
            counts.append(entry.count(image_token))
        elif isinstance(entry, (list, tuple)):
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
        """Adopt ``dynamic_vision`` from a loaded Qwen3-VL model's vision MXQ.

        In the typical flow, ``dynamic_vision`` is populated automatically by
        :meth:`from_pretrained` reading the top-level ``config.dynamic_vision``
        from the shipped ``config.json``, so calling this helper is
        unnecessary.

        Use it only when the processor and model have diverged at runtime —
        e.g. the model was loaded with an explicit ``vision_mxq_path=`` kwarg
        pointing at an MXQ whose signature (static/dynamic) doesn't match the
        config the processor was built from.

        The vision submodule detects its own signature from
        ``get_input_buffer_info()`` and stores the result on
        ``visual._uses_dynamic_vision``. Calling this helper unconditionally
        overwrites ``dynamic_vision`` (and the video processor's mirror) with
        that detected value, so the processor's resize / max-pixel clamp
        stays in lock-step with what the compiled model can actually consume.
        Any prior value — the class default, a config-derived assignment from
        :meth:`from_pretrained`, or an explicit user override — is replaced.
        """
        vision = getattr(getattr(model, "model", model), "visual", None)
        if vision is None or not hasattr(vision, "_uses_dynamic_vision"):
            raise ValueError(
                "sync_dynamic_vision_from_model expects a Qwen3-VL model whose "
                "vision submodule exposes `_uses_dynamic_vision`."
            )
        self.dynamic_vision = bool(vision._uses_dynamic_vision)
        self._sync_dynamic_vision_to_video_processor()

    @staticmethod
    def _resize_one(img, size=(224, 224)):
        if isinstance(img, str):
            img = load_image(img)
        if isinstance(img, Image.Image):
            return img.resize(size)
        if isinstance(img, np.ndarray):
            # A 4-D batch is a valid ``ImageInput`` shape that the upstream
            # Qwen3-VL processor unrolls into per-frame images before its own
            # resize; ``_count_images`` accepts both ``(N, H, W, C)`` and
            # ``(N, C, H, W)`` layouts, so both must survive here. ``cv2.resize``
            # only handles a single 2-D or 3-D HWC array, so split along the batch
            # axis, detect each frame's channel layout, transpose channels-first
            # frames to HWC for the resize, and restore the original layout on the
            # way out. When both endpoints look like plausible channel counts
            # (e.g. a small square image where ``shape[0] == shape[-1] == 3``),
            # tie-break to HWC to match ``_count_images`` and the majority
            # upstream convention.
            if img.ndim == 4:
                resized_frames = []
                for frame in img:
                    first, last = frame.shape[0], frame.shape[-1]
                    channels_last = last in (1, 3, 4)
                    channels_first = first in (1, 3, 4)
                    if channels_first and not channels_last:
                        hwc = np.transpose(frame, (1, 2, 0))
                        resized_hwc = cast(
                            np.ndarray,
                            cv2_resize(hwc, size[::-1], interpolation=INTER_CUBIC),
                        )
                        resized_frames.append(np.transpose(resized_hwc, (2, 0, 1)))
                    else:
                        resized_frames.append(
                            cast(
                                np.ndarray,
                                cv2_resize(frame, size[::-1], interpolation=INTER_CUBIC),
                            )
                        )
                return np.stack(resized_frames)
            return cast(np.ndarray, cv2_resize(img, size[::-1], interpolation=INTER_CUBIC))
        if torch.is_tensor(img):
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.ndim == 3:
                img = img.unsqueeze(0)
            return F.interpolate(img.float(), size=size, mode="bicubic", align_corners=False)
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
        """
        ip = self.image_processor
        limit = self.max_vision_tokens * ip.patch_size ** 2
        current_longest = ip.size["longest_edge"]
        if current_longest <= limit:
            return

        logger.info(
            "[dynamic-vision] capped max_pixels %d -> %d (<= %d vision tokens)",
            current_longest,
            limit,
            self.max_vision_tokens,
        )
        ip.size = _update_size(
            ip.size,
            longest_edge=limit,
            shortest_edge=min(ip.size["shortest_edge"], limit),
        )

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
        """
        vp = self.video_processor
        if vp is None:
            return
        limit = self.max_vision_tokens * vp.patch_size ** 2 * vp.temporal_patch_size
        current_longest = vp.size["longest_edge"]
        if current_longest <= limit:
            return

        logger.info(
            "[dynamic-vision] capped video max_pixels %d -> %d (<= %d vision tokens/frame)",
            current_longest,
            limit,
            self.max_vision_tokens,
        )
        vp.size = _update_size(
            vp.size,
            longest_edge=limit,
            shortest_edge=min(vp.size["shortest_edge"], limit),
        )

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

    def _reject_do_resize_false(self, scope: dict, kind: str) -> None:
        if scope.get("do_resize") is False:
            raise ValueError(
                f"do_resize=False bypasses the {self.max_vision_tokens}-token NPU "
                f"vision-token ceiling for {kind} inputs. Larger inputs hang the "
                "NPU (watchdog timeout -> Model_NotAlive) rather than erroring "
                "cleanly. Remove the do_resize override or pre-resize the input "
                "so its resulting patch grid fits the ceiling."
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

        if images is not None:
            if self.dynamic_vision:
                # Keep the aspect ratio, but stay inside the NPU sequence limit.
                # Two-stage clamp: cap the stored defaults, then re-clamp any
                # caller overrides so ``size`` / ``max_pixels`` / ``min_pixels``
                # / ``do_resize`` cannot bypass the ceiling. See
                # ``_clamp_dynamic_image_call_kwargs`` for the override contract.
                self._clamp_dynamic_image_size()
                self._clamp_dynamic_image_call_kwargs(kwargs)
                logger.debug("[dynamic-vision] skipping forced resize, keeping original aspect ratio")
            else:
                # Static Qwen3-VL MXQ releases bake a single image's 2D RoPE grid into
                # the text decoder. A second image *for the same prompt* would need
                # its own independent 2D coordinates, which the baked rope cannot
                # express — the decoder loses the image-boundary distinction and
                # emits grammatically-plausible but semantically wrong output. Fail
                # here, before ``_resize_images`` and the image_processor's patch
                # extraction, with the same shape of message the video hard-fail
                # uses. The guard is per-prompt (a batch of N single-image prompts
                # must pass through), so match upstream's association order and
                # count ``<|image_pad|>`` placeholders per prompt rather than
                # inferring per-prompt counts from the ``images`` container shape.
                image_token = getattr(self, "image_token", _IMAGE_PAD_TOKEN)
                per_prompt_counts = _per_prompt_image_pad_counts(text, image_token)
                if per_prompt_counts and max(per_prompt_counts) > 1:
                    raise NotImplementedError(
                        "Multi-image input requires a dynamic-vision Qwen3-VL release "
                        "(3-input vision MXQ with per-image 2D RoPE in the text "
                        "decoder). The currently loaded processor is in static mode "
                        "(dynamic_vision=False), which supports exactly one image per "
                        "prompt. Load a Qwen3-VL release that ships a dynamic vision "
                        "MXQ, or pass a single image."
                    )
                images = self._resize_images(images)

        if videos is not None:
            # Static Qwen3-VL MXQ releases (single-input vision, fixed visual-token count in the
            # text decoder) cannot express video: per-frame RoPE and variable-length visual
            # regions are exactly what the dynamic vision MXQ was compiled to carry. Without
            # that, video decoding + preprocessing would still run and the language model
            # would emit grammatically-plausible but semantically empty output. Fail here —
            # before the heavy torchcodec/FFmpeg video decode — with a message pointing to
            # the release that actually supports video.
            if not self.dynamic_vision:
                raise NotImplementedError(
                    "Video input requires a dynamic-vision Qwen3-VL release (3-input vision "
                    "MXQ with variable visual-token count in the text decoder). The currently "
                    "loaded processor is in static mode (dynamic_vision=False). Load a "
                    "Qwen3-VL release that ships a dynamic vision MXQ, or pass only image "
                    "inputs."
                )
            self._sync_dynamic_vision_to_video_processor()
            # Keep the aspect ratio, but stay inside the NPU per-frame sequence
            # limit. Two-stage clamp: cap the stored defaults, then re-clamp
            # any caller overrides (``size``, ``do_resize``) so they cannot
            # bypass the ceiling. See ``_clamp_dynamic_video_call_kwargs`` for
            # the override contract.
            self._clamp_dynamic_video_size()
            self._clamp_dynamic_video_call_kwargs(kwargs)
            text = self._strip_video_outer_wrap(text)

        # transformers 5.x's ``Qwen3VLModel.compute_3d_position_ids`` (and the
        # generate-side ``_prepare_position_ids_for_generation``) build MRoPE
        # 3-D t/h/w positions only when ``mm_token_type_ids`` is present.
        # Without it, both fall back to linear (non-MRoPE) positions and the
        # decoder cannot distinguish visual tokens by time/space, producing
        # degenerate output on video inputs. tf 4.x has neither hook and its
        # ``generate`` strictly rejects unknown model_kwargs, so populating
        # ``mm_token_type_ids`` unconditionally raises there — gate on the
        # ``create_mm_token_type_ids`` method that tf 5.x introduced.
        if (images is not None or videos is not None) and hasattr(
            self, "create_mm_token_type_ids"
        ):
            text_kwargs = kwargs.get("text_kwargs")
            if text_kwargs is None:
                text_kwargs = {}
                kwargs["text_kwargs"] = text_kwargs
            text_kwargs.setdefault("return_mm_token_type_ids", True)

        return super().__call__(images, text, videos, **kwargs)


AutoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLProcessor)
AutoVideoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLVideoProcessor)
