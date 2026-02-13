from typing import List

import cv2
import torch

from ..types import TensorLike
from .base import PreOps


class LetterBox(PreOps):
    """Preprocessing for YOLO models, implementing letterbox resizing.

    Resizes the image while maintaining aspect ratio, adding padding to meet
    target dimensions. Based on Ultralytics implementation.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    """

    def __init__(self, img_size: List[int]):
        """Initializes LetterBox with target image size.

        Args:
            img_size (List[int]): Target image size [h, w].
        """
        super().__init__()
        self.img_size = img_size

    def __call__(self, x: TensorLike) -> torch.Tensor:
        """Executes YOLO preprocessing (letterbox resizing).

        Args:
            x (TensorLike): Input image.

        Returns:
            torch.Tensor: Preprocessed image in HWC format on the selected device.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        img = x
        h0, w0 = img.shape[:2]  # original hw
        r = min(self.img_size[0] / h0, self.img_size[1] / w0)  # ratio
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dh, dw = (
            self.img_size[0] - new_unpad[1],
            self.img_size[1] - new_unpad[0],
        )  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if (img.shape[1], img.shape[0]) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        return torch.from_numpy(img).to(self.device).byte()
