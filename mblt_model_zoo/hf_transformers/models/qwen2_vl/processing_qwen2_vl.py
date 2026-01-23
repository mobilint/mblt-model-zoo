from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import INTER_CUBIC
from cv2 import resize as cv2_resize
from PIL import Image
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, load_image
from transformers.models.qwen2_vl.processing_qwen2_vl import (
    Qwen2VLProcessor,
    Qwen2VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from .configuration_qwen2_vl import MobilintQwen2VLConfig


class MobilintQwen2VLProcessor(Qwen2VLProcessor):
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        assert text is not None, "text is None!"
        
        # Make sure images is only one instance of PIL.Image.Image, np.ndarray, torch.Tensor, or None
        while isinstance(images, list):
            if len(images) > 1:
                raise NotImplementedError("Only one image input is supported")
            images = images[0]
        
        if isinstance(images, str):
            images = load_image(images)
        
        # Image should be resized into (224, 224) to fit image token position
        size = (224, 224)
        
        if isinstance(images, Image.Image):
            images = images.resize(size)
        elif isinstance(images, np.ndarray):
            if images.ndim == 2:  # greyscale
                return cast(BatchFeature, cv2_resize(images, size[::-1], interpolation=INTER_CUBIC))
            elif images.ndim == 3:
                return cast(BatchFeature, cv2_resize(images, size[::-1], interpolation=INTER_CUBIC))
            else:
                raise ValueError(f"Unsupported ndarray shape: {images.shape}")
        elif torch.is_tensor(images):
            if images.ndim == 3:  # CHW
                images = images.unsqueeze(0).float()  # BCHW
                images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
            elif images.ndim == 2:  # HW
                images = images.unsqueeze(0).unsqueeze(0).float()  # B1HW
                images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(images.shape)}")
        else:
            raise TypeError(f"Unsupported type of image: {type(images)}")
        
        if videos is not None:
            raise NotImplementedError("Video inputs are not supported")
                    
        return super().__call__(images, text, videos, **kwargs)

AutoProcessor.register(MobilintQwen2VLConfig, MobilintQwen2VLProcessor)
