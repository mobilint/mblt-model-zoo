from typing import Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, load_image
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor, Qwen2VLProcessorKwargs
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from .configuration_qwen2_vl import MobilintQwen2VLConfig


class MobilintQwen2VLProcessor(Qwen2VLProcessor):
    """Qwen2-VL processor with a compiled-graph-driven dynamic path."""

    dynamic_vision = False
    max_vision_tokens = 2048

    def __init__(self, *args, dynamic_vision: bool = False, max_vision_tokens: int = 2048, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_vision = bool(dynamic_vision)
        self.max_vision_tokens = int(max_vision_tokens)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if not isinstance(processor, cls):
            return processor
        config_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "subfolder", "token", "trust_remote_code")
            if key in kwargs
        }
        try:
            config = MobilintQwen2VLConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        except (OSError, ValueError, KeyError):
            return processor
        processor.dynamic_vision = bool(getattr(config, "dynamic_vision", False))
        processor.max_vision_tokens = int(getattr(config, "max_vision_tokens", 2048))
        return processor

    @staticmethod
    def _cap_scope(scope: dict, limit: int) -> None:
        for key in ("max_pixels", "min_pixels"):
            if scope.get(key) is not None:
                scope[key] = min(int(scope[key]), limit)
        size = scope.get("size")
        if isinstance(size, dict):
            for key in ("longest_edge", "shortest_edge"):
                if size.get(key) is not None:
                    size[key] = min(int(size[key]), limit)

    def _clamp_dynamic_kwargs(self, kwargs: dict, *, patch_area: int, nested_key: str) -> None:
        """Limit smart_resize's pixel budget so the merged patch count stays bounded."""
        pixel_limit = max(1, int(self.max_vision_tokens)) * patch_area
        scopes = [kwargs]
        nested = kwargs.get(nested_key)
        if isinstance(nested, dict):
            scopes.append(nested)
        for scope in scopes:
            self._cap_scope(scope, pixel_limit)
        processor = self.image_processor if nested_key == "images_kwargs" else self.video_processor
        if processor is not None and hasattr(processor, "max_pixels"):
            processor.max_pixels = min(int(processor.max_pixels), pixel_limit)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("text is required for Qwen2-VL processing")
        if self.dynamic_vision:
            patch_area = (int(self.image_processor.patch_size) * int(self.image_processor.merge_size)) ** 2
            if images is not None:
                self._clamp_dynamic_kwargs(kwargs, patch_area=patch_area, nested_key="images_kwargs")
            if videos is not None:
                self._clamp_dynamic_kwargs(kwargs, patch_area=patch_area, nested_key="videos_kwargs")
            return super().__call__(images, text, videos, **kwargs)

        if videos is not None:
            raise NotImplementedError("Video inputs require a dynamic-vision Qwen2-VL release")
        while isinstance(images, list):
            if len(images) > 1:
                raise NotImplementedError("Only one image input is supported by static Qwen2-VL")
            images = images[0] if images else None
        if isinstance(images, str):
            images = load_image(images)
        if images is not None:
            from PIL import Image
            import numpy as np
            import torch
            import torch.nn.functional as F

            size = (224, 224)
            if isinstance(images, Image.Image):
                images = images.resize(size)
            elif isinstance(images, np.ndarray):
                tensor = torch.from_numpy(images)
                if tensor.ndim == 2:
                    tensor = tensor[None, None]
                elif tensor.ndim == 3:
                    tensor = tensor.permute(2, 0, 1)[None] if tensor.shape[-1] in (1, 3) else tensor[None]
                else:
                    raise ValueError(f"Unsupported ndarray shape: {images.shape}")
                images = F.interpolate(tensor.float(), size=size, mode="bicubic", align_corners=False).squeeze(0)
            elif torch.is_tensor(images):
                if images.ndim == 2:
                    images = images[None, None]
                elif images.ndim == 3:
                    images = images[None] if images.shape[0] in (1, 3) else images.permute(2, 0, 1)[None]
                elif images.ndim != 4:
                    raise ValueError(f"Unsupported tensor shape: {tuple(images.shape)}")
                images = F.interpolate(images.float(), size=size, mode="bicubic", align_corners=False)
            else:
                raise TypeError(f"Unsupported type of image: {type(images)}")
        return super().__call__(images, text, videos, **kwargs)


AutoProcessor.register(MobilintQwen2VLConfig, MobilintQwen2VLProcessor)
