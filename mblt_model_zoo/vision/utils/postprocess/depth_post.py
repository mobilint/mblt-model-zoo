"""Postprocessing for monocular depth-estimation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from ..types import ListTensorLike, TensorLike
from ._letterbox import crop_letterbox, get_letterbox_input_shape, normalize_ratio_pads, normalize_shapes
from .base import PostBase
from .common import RatioPad


class DepthPost(PostBase):
    """Normalize depth outputs and undo letterbox padding when metadata is available."""

    def __init__(self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any]) -> None:
        """Initialize depth restoration from the model letterbox configuration."""

        super().__init__()
        del post_cfg
        self.input_shape = get_letterbox_input_shape(pre_cfg, "Depth estimation", "Depth")

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

        shapes = normalize_shapes(img0_shape, depth.shape[0])
        pads = normalize_ratio_pads(ratio_pad, depth.shape[0], shapes, self.input_shape)
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

    def _restore(self, depth: torch.Tensor, shape: tuple[int, int], ratio_pad: RatioPad) -> torch.Tensor:
        """Crop padded depth pixels and bilinearly resize to an original image shape."""

        cropped = crop_letterbox(depth, shape, ratio_pad, self.input_shape, "Depth")
        return functional.interpolate(cropped[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
