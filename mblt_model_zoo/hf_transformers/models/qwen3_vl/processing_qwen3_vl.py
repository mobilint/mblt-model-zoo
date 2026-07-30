from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import INTER_CUBIC
from cv2 import resize as cv2_resize
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, load_image
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.generic import logging
from transformers.video_utils import VideoInput

from .configuration_qwen3_vl import MobilintQwen3VLConfig

logger = logging.get_logger(__name__)

# NPU vision model fixed input shape: (H_npu, W_npu, C_npu) = (1024, 64, 6)
_NPU_H, _NPU_W = 1024, 64

# The dynamic vision MXQ takes the pre-merge patch sequence as `inputs[0]`. Its op
# descriptor declares a 4096-token ceiling, but anything above 2048 hangs the NPU
# (watchdog timeout -> `Model_NotAlive`) rather than erroring out, so the default is
# the largest length measured to run. Override `max_vision_tokens` for an MXQ that
# supports longer sequences.
_NPU_MAX_VISION_TOKENS = 2048


def _compute_npu_frame_size(patch_size: int, merge_size: int) -> tuple[int, int]:
    """Derive the pixel resolution that produces the NPU-compatible grid."""
    pw = _NPU_W // (merge_size ** 2)
    gh_merged = int((_NPU_H // pw) ** 0.5)
    side = gh_merged * merge_size * patch_size
    return (side, side)


class MobilintQwen3VLProcessor(Qwen3VLProcessor):
    dynamic_vision = False
    max_vision_tokens = _NPU_MAX_VISION_TOKENS

    def sync_dynamic_vision_from_model(self, model) -> None:
        """Adopt ``dynamic_vision`` from a loaded Qwen3-VL model's vision MXQ.

        The vision submodule detects its own signature from
        ``get_input_buffer_info()`` and stores the result on
        ``visual._uses_dynamic_vision``. Mirror that here so the processor's
        resize / max-pixel clamp stays in lock-step with what the compiled
        model can actually consume.

        If the processor was previously assigned an explicit
        ``dynamic_vision`` that disagrees with the model's detection this
        raises rather than silently overriding — a silent mismatch would let
        the processor emit tensors the compiled model can't consume, which
        is the exact footgun this helper exists to prevent.
        """
        vision = getattr(getattr(model, "model", model), "visual", None)
        if vision is None or not hasattr(vision, "_uses_dynamic_vision"):
            raise ValueError(
                "sync_dynamic_vision_from_model expects a Qwen3-VL model whose "
                "vision submodule exposes `_uses_dynamic_vision`."
            )
        detected = bool(vision._uses_dynamic_vision)
        # Only an explicit instance-level assignment counts as a "user
        # override" worth guarding: the class-level default of False must
        # transparently upgrade to True when the model is dynamic, otherwise
        # nobody would benefit from calling this helper.
        if "dynamic_vision" in self.__dict__ and bool(self.dynamic_vision) != detected:
            raise ValueError(
                f"Processor.dynamic_vision={self.dynamic_vision} conflicts with "
                f"model MXQ detection ({detected}). Reset processor.dynamic_vision "
                "or reload with a matching MXQ."
            )
        self.dynamic_vision = detected

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

    def _install_video_resize_hook(self) -> None:
        """Override video_processor._preprocess to force NPU-compatible frame size."""
        vp = self.video_processor
        if getattr(vp, "_mobilint_hooked", False):
            return

        target = _compute_npu_frame_size(vp.patch_size, vp.merge_size)
        orig_preprocess = vp._preprocess

        def _hooked_preprocess(videos, do_resize=True, size=None, **kw):
            resized = []
            for v in videos:
                T, C, H, W = v.shape
                resized.append(F.interpolate(v.float(), size=target, mode="bicubic", align_corners=False))
            return orig_preprocess(resized, do_resize=False, size=size, **kw)

        vp._preprocess = _hooked_preprocess
        vp._mobilint_hooked = True

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
        if ip.size["longest_edge"] <= limit:
            return

        logger.info(
            "[dynamic-vision] capped max_pixels %d -> %d (<= %d vision tokens)",
            ip.size["longest_edge"],
            limit,
            self.max_vision_tokens,
        )
        ip.size = {
            **ip.size,
            "longest_edge": limit,
            "shortest_edge": min(ip.size["shortest_edge"], limit),
        }

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
                images = self._resize_images(images)

        if videos is not None:
            self._install_video_resize_hook()

        return super().__call__(images, text, videos, **kwargs)


AutoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLProcessor)
