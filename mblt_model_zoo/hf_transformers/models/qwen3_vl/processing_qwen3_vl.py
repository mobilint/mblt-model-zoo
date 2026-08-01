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
_NPU_MAX_VISION_TOKENS = 2048


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


def _max_images_per_prompt(images) -> int:
    """Return the largest image count across prompts.

    The chat-template pipeline wraps images as ``[[imgs_prompt_1], [imgs_prompt_2], ...]``
    (outer list = prompts, inner list = images per prompt), so a batch of N
    single-image prompts arrives as ``[[img_1], [img_2], ..., [img_N]]`` — a
    total count of N that must not trip the static multi-image hard-fail. The
    static MXQ constraint is "at most one image per prompt", so we return the
    max per-prompt count; ``_count_images`` totals across all prompts and is
    the wrong quantity for the guard. A flat list (single prompt with
    multiple images) or a bare image is treated as one prompt.
    """
    if images is None:
        return 0
    if (
        isinstance(images, (list, tuple))
        and images
        and all(isinstance(item, (list, tuple)) for item in images)
    ):
        return max((_count_images(item) for item in images), default=0)
    return _count_images(images)


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
        config_kwargs = {
            k: kwargs[k]
            for k in ("revision", "cache_dir", "token", "trust_remote_code")
            if k in kwargs
        }
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
            vision_dyn = bool(getattr(config, "dynamic_vision", False))
        except Exception as exc:
            logger.debug(
                "Falling back to processor default dynamic_vision (config load failed: %s)",
                exc,
            )
            vision_dyn = False
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
                self._clamp_dynamic_image_size()
                logger.debug("[dynamic-vision] skipping forced resize, keeping original aspect ratio")
            else:
                # Static Qwen3-VL MXQ releases bake a single image's 2D RoPE grid into
                # the text decoder. A second image would need its own independent 2D
                # coordinates, which the baked rope cannot express — the decoder loses
                # the image-boundary distinction and emits grammatically-plausible but
                # semantically wrong output. Fail here, before ``_resize_images`` and
                # the image_processor's patch extraction, with the same shape of
                # message the video hard-fail uses.
                if _max_images_per_prompt(images) > 1:
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
