"""Postprocessing for monocular depth-estimation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from ..types import ListTensorLike, TensorLike
from ._letterbox import crop_letterbox, get_letterbox_input_shape, resolve_ratio_pads
from .base import PostBase
from .common import RatioPad, normalize_image_shapes


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

        shapes = normalize_image_shapes(img0_shape, depth.shape[0])
        pads = resolve_ratio_pads(ratio_pad, depth.shape[0], shapes, self.input_shape)
        restored = [self._restore(depth[index], shapes[index], pads[index]) for index in range(depth.shape[0])]
        return restored[0] if len(restored) == 1 else restored

    def _normalize_output(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Validate dense output and normalize it to input-sized ``[B, H, W]`` tensors."""

        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"Depth estimation expects one output tensor, received {len(x)}.")
            x = x[0]
        depth = torch.as_tensor(x)
        if depth.ndim == 4:
            if depth.shape[1] == 1:
                depth = depth[:, 0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                raise ValueError(
                    f"Depth estimation expects [B, 1, H, W] or [B, H, W, 1], got {tuple(depth.shape)}."
                )
        elif depth.ndim == 3 and depth.shape[-1] == 1 and tuple(depth.shape[:2]) == self.input_shape:
            depth = depth[..., 0].unsqueeze(0)
        elif depth.ndim != 3:
            raise ValueError(
                "Depth estimation expects [B, H, W], [B, 1, H, W], [H, W, 1], or [B, H, W, 1], "
                f"got {tuple(depth.shape)}."
            )
        depth = depth.to(device=self.device, dtype=torch.float32)
        if tuple(depth.shape[-2:]) == self.input_shape:
            return depth

        quarter_shape = tuple(dimension // 4 for dimension in self.input_shape)
        if tuple(depth.shape[-2:]) == quarter_shape:
            return functional.interpolate(depth[:, None], scale_factor=4.0, mode="bilinear", align_corners=False)[:, 0]

        raise ValueError(
            f"Depth estimation output spatial shape must be {self.input_shape} or quarter-resolution {quarter_shape}, "
            f"got {tuple(depth.shape[-2:])}."
        )

    def _restore(self, depth: torch.Tensor, shape: tuple[int, int], ratio_pad: RatioPad) -> torch.Tensor:
        """Crop padded depth pixels and bilinearly resize to an original image shape."""

        cropped = crop_letterbox(depth, shape, ratio_pad, self.input_shape, "Depth")
        return functional.interpolate(cropped[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
