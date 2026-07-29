from __future__ import annotations

import cv2
import numpy as np
import torch

from ..letterbox import LetterBoxGeometry, RatioPad
from ..types import TensorLike
from .base import PreOps


def _apply_letterbox(
    image: np.ndarray,
    img_size: list[int],
    interpolation: int,
    padding_value: int | tuple[int, int, int],
) -> tuple[np.ndarray, RatioPad]:
    """Resize and pad an array while preserving its aspect ratio.

    Args:
        image: Image or two-dimensional semantic mask.
        img_size: Target size as ``[height, width]``.
        interpolation: OpenCV interpolation mode.
        padding_value: Constant border value.

    Returns:
        The letterboxed array and its resize/padding metadata.
    """

    input_shape = (int(img_size[0]), int(img_size[1]))
    original_shape = (int(image.shape[0]), int(image.shape[1]))
    geometry = LetterBoxGeometry.from_shapes(input_shape, original_shape)
    resized_height, resized_width = geometry.resized_shape
    if image.shape[:2] != geometry.resized_shape:
        image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    top, bottom, left, right = geometry.borders
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )
    return image, geometry.ratio_pad


def letterbox_semantic_mask(
    mask: np.ndarray,
    img_size: list[int],
    ignore_label: int = 255,
) -> tuple[np.ndarray, RatioPad]:
    """Letterbox a semantic mask without interpolating class IDs.

    Args:
        mask: Two-dimensional semantic class map.
        img_size: Target size as ``[height, width]``.
        ignore_label: Class value used for padded pixels.

    Returns:
        The letterboxed mask and its resize/padding metadata.

    Raises:
        ValueError: If the mask is not two-dimensional.
    """

    if mask.ndim != 2:
        raise ValueError(f"Semantic masks must be two-dimensional, got shape {mask.shape}.")
    return _apply_letterbox(mask, img_size, cv2.INTER_NEAREST, ignore_label)


class LetterBox(PreOps):
    """Preprocessing for YOLO models, implementing letterbox resizing.

    Resizes the image while maintaining aspect ratio, adding padding to meet
    target dimensions. Based on Ultralytics implementation.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    """

    def __init__(self, img_size: list[int]) -> None:
        """Initializes LetterBox with target image size.

        Args:
            img_size (list[int]): Target image size [h, w].
        """
        super().__init__()
        self.img_size = img_size
        self.ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __call__(self, x: TensorLike) -> torch.Tensor:
        """Executes YOLO preprocessing (letterbox resizing).

        Args:
            x (TensorLike): Input image.

        Returns:
            torch.Tensor: Preprocessed image in HWC format on the selected device.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        img, self.ratio_pad = _apply_letterbox(
            x,
            self.img_size,
            cv2.INTER_LINEAR,
            (114, 114, 114),
        )
        return torch.from_numpy(img).to(self.device).byte()
