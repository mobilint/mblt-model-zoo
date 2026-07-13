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
from transformers.video_utils import VideoInput

from .configuration_qwen3_vl import MobilintQwen3VLConfig

# NPU vision model fixed input shape: (H_npu, W_npu, C_npu) = (1024, 64, 6)
_NPU_H, _NPU_W = 1024, 64


def _compute_npu_frame_size(patch_size: int, merge_size: int) -> tuple[int, int]:
    """Derive the pixel resolution that produces the NPU-compatible grid."""
    pw = _NPU_W // (merge_size ** 2)
    gh_merged = int((_NPU_H // pw) ** 0.5)
    side = gh_merged * merge_size * patch_size
    return (side, side)


class MobilintQwen3VLProcessor(Qwen3VLProcessor):
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

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        assert text is not None, "text is None!"

        if images is not None:
            images = self._resize_images(images)

        if videos is not None:
            self._install_video_resize_hook()

        return super().__call__(images, text, videos, **kwargs)


AutoProcessor.register(MobilintQwen3VLConfig, MobilintQwen3VLProcessor)
