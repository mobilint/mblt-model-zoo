"""Postprocessing for monocular depth-estimation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as functional

from ..types import ListTensorLike, TensorLike
from .base import PostBase
from .common import RatioPad


class DepthPost(PostBase):
    """Normalize depth outputs and undo letterbox padding when metadata is available."""

    def __init__(self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any]) -> None:
        """Initialize depth restoration from the model letterbox configuration."""

        super().__init__()
        del post_cfg
        letterbox_cfg = pre_cfg.get("LetterBox")
        if not isinstance(letterbox_cfg, dict) or "img_size" not in letterbox_cfg:
            raise ValueError("Depth estimation requires a LetterBox configuration in pre_cfg.")
        image_size = letterbox_cfg["img_size"]
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise ValueError("Depth LetterBox img_size must be a two-item [height, width] list.")
        self.input_shape = (int(image_size[0]), int(image_size[1]))

    def __call__(
        self,
        x: TensorLike | ListTensorLike,
        img0_shape: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
        ratio_pad: RatioPad | Sequence[RatioPad | None] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return normalized depth maps, optionally restored to original image sizes."""

        if kwargs:
            raise TypeError(f"Unexpected depth postprocess kwargs: {', '.join(sorted(kwargs))}")
        depth = self._normalize_output(x)
        if img0_shape is None:
            return depth

        shapes = self._as_shapes(img0_shape, depth.shape[0])
        pads = self._as_ratio_pads(ratio_pad, depth.shape[0], shapes)
        restored = [self._restore(depth[index], shapes[index], pads[index]) for index in range(depth.shape[0])]
        return restored[0] if len(restored) == 1 else restored

    def _normalize_output(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Validate and normalize ONNX output to ``[B, H, W]`` tensors."""

        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"Depth estimation expects one output tensor, received {len(x)}.")
            x = x[0]
        depth = torch.as_tensor(x)
        if depth.ndim == 4:
            if depth.shape[1] != 1:
                raise ValueError(f"Depth estimation expects [B, 1, H, W], got {tuple(depth.shape)}.")
            depth = depth[:, 0]
        elif depth.ndim != 3:
            raise ValueError(f"Depth estimation expects [B, H, W] or [B, 1, H, W], got {tuple(depth.shape)}.")
        return depth.to(device=self.device, dtype=torch.float32)

    def _as_shapes(
        self, img0_shape: tuple[int, int] | Sequence[tuple[int, int]], batch_size: int
    ) -> list[tuple[int, int]]:
        """Normalize one or many original image shapes to a batch-sized list."""

        if len(img0_shape) == 2 and isinstance(img0_shape[0], int):
            return [(int(img0_shape[0]), int(img0_shape[1]))] * batch_size  # type: ignore[index]
        shapes = [(int(shape[0]), int(shape[1])) for shape in img0_shape]  # type: ignore[union-attr]
        if len(shapes) != batch_size:
            raise ValueError(f"Expected {batch_size} original image shapes, got {len(shapes)}.")
        return shapes

    def _as_ratio_pads(
        self,
        ratio_pad: RatioPad | Sequence[RatioPad | None] | None,
        batch_size: int,
        shapes: Sequence[tuple[int, int]],
    ) -> list[RatioPad]:
        """Normalize explicit or derived letterbox metadata for every sample."""

        if ratio_pad is None:
            return [self._ratio_pad_for_shape(shape) for shape in shapes]
        if isinstance(ratio_pad, tuple) and len(ratio_pad) == 2 and isinstance(ratio_pad[0], tuple):
            return [cast(RatioPad, ratio_pad)] * batch_size
        pads: list[RatioPad] = [
            cast(RatioPad, pad) if pad is not None else self._ratio_pad_for_shape(shapes[index])
            for index, pad in enumerate(ratio_pad)
        ]
        if len(pads) != batch_size:
            raise ValueError(f"Expected {batch_size} ratio_pad values, got {len(pads)}.")
        return pads

    def _ratio_pad_for_shape(self, shape: tuple[int, int]) -> RatioPad:
        """Recreate Ultralytics-compatible letterbox metadata for an original shape."""

        height, width = shape
        ratio = min(self.input_shape[0] / height, self.input_shape[1] / width)
        unpadded_width, unpadded_height = int(round(width * ratio)), int(round(height * ratio))
        pad_x = int(round((self.input_shape[1] - unpadded_width) / 2 - 0.1))
        pad_y = int(round((self.input_shape[0] - unpadded_height) / 2 - 0.1))
        return ((ratio, ratio), (pad_x, pad_y))

    def _restore(self, depth: torch.Tensor, shape: tuple[int, int], ratio_pad: RatioPad) -> torch.Tensor:
        """Crop padded depth pixels and bilinearly resize to an original image shape."""

        ratio, pad = ratio_pad
        del ratio
        output_height, output_width = depth.shape
        scale_x = output_width / self.input_shape[1]
        scale_y = output_height / self.input_shape[0]
        left = int(round(pad[0] * scale_x))
        top = int(round(pad[1] * scale_y))
        original_height, original_width = shape
        resize_ratio = min(self.input_shape[0] / original_height, self.input_shape[1] / original_width)
        unpadded_width = int(round(original_width * resize_ratio * scale_x))
        unpadded_height = int(round(original_height * resize_ratio * scale_y))
        cropped = depth[top : top + unpadded_height, left : left + unpadded_width]
        if cropped.numel() == 0:
            raise ValueError("Depth letterbox restoration produced an empty crop.")
        return functional.interpolate(cropped[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
