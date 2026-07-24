"""Postprocessing for semantic-segmentation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from ..types import ListTensorLike, TensorLike
from ._letterbox import crop_letterbox, get_letterbox_input_shape, normalize_ratio_pads, normalize_shapes
from .base import PostBase
from .common import RatioPad


class SemanticSegPost(PostBase):
    """Convert semantic logits to class maps and undo letterbox padding."""

    NC_BY_DATASET: dict[str, int] = {
        "ade20k": 150,
        "cityscapes": 19,
    }

    def __init__(self, pre_cfg: dict[str, Any], post_cfg: dict[str, Any]) -> None:
        """Initialize semantic output handling for a configured dataset taxonomy."""

        super().__init__()
        # Preserve the validation messages previously inherited from DepthPost.
        self.input_shape = get_letterbox_input_shape(pre_cfg, "Depth estimation", "Depth")
        dataset = post_cfg.get("dataset")
        if not isinstance(dataset, str):
            raise ValueError("Semantic segmentation requires a string dataset in post_cfg.")
        self.dataset = dataset.lower()
        dataset_nc = self.NC_BY_DATASET.get(self.dataset)
        configured_nc = post_cfg.get("nc")
        if configured_nc is None:
            if dataset_nc is None:
                raise ValueError(f"Semantic segmentation requires nc for unknown dataset '{self.dataset}'.")
            self.nc = dataset_nc
        else:
            self.nc = int(configured_nc)
        if dataset_nc is not None and self.nc != dataset_nc:
            raise ValueError(
                f"nc={configured_nc} conflicts with semantic dataset '{self.dataset}', which requires nc={dataset_nc}."
            )

    def __call__(
        self,
        x: TensorLike | ListTensorLike,
        img0_shape: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
        ratio_pad: RatioPad | Sequence[RatioPad | None] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return class maps, optionally restored to original image sizes."""

        if kwargs:
            raise TypeError(f"Unexpected semantic postprocess kwargs: {', '.join(sorted(kwargs))}")
        class_maps = self._normalize_output(x)
        if img0_shape is None:
            return class_maps

        shapes = normalize_shapes(img0_shape, class_maps.shape[0])
        pads = normalize_ratio_pads(ratio_pad, class_maps.shape[0], shapes, self.input_shape)
        restored = [
            self._restore(class_maps[index], shapes[index], pads[index]) for index in range(class_maps.shape[0])
        ]
        return restored[0] if len(restored) == 1 else restored

    def _normalize_output(self, x: TensorLike | ListTensorLike) -> torch.Tensor:
        """Normalize logits or baked maps to integer ``[B, H, W]`` class maps."""

        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"Semantic segmentation expects one output tensor, received {len(x)}.")
            x = x[0]
        output = torch.as_tensor(x, device=self.device)
        if output.ndim == 4:
            if output.shape[1] != self.nc:
                raise ValueError(
                    f"Semantic segmentation for '{self.dataset}' expects [B, {self.nc}, H, W] logits, "
                    f"got {tuple(output.shape)}."
                )
            logits = output.to(dtype=torch.float32)
            if tuple(logits.shape[-2:]) != self.input_shape:
                logits = functional.interpolate(logits, size=self.input_shape, mode="bilinear", align_corners=False)
            return logits.argmax(dim=1).to(dtype=torch.int64)
        if output.ndim == 3:
            class_map = output.to(dtype=torch.float32)
            if tuple(class_map.shape[-2:]) != self.input_shape:
                class_map = functional.interpolate(class_map[:, None], size=self.input_shape, mode="nearest")[:, 0]
            class_map = class_map.to(dtype=torch.int64)
            if class_map.numel() and (int(class_map.min()) < 0 or int(class_map.max()) >= self.nc):
                raise ValueError(f"Semantic class-map values must be in [0, {self.nc - 1}].")
            return class_map
        raise ValueError(
            f"Semantic segmentation expects [B, C, H, W] logits or [B, H, W] class maps, got {tuple(output.shape)}."
        )

    def _restore(self, class_map: torch.Tensor, shape: tuple[int, int], ratio_pad: RatioPad) -> torch.Tensor:
        """Crop padded pixels and nearest-resize a class map to its original shape."""

        cropped = crop_letterbox(class_map, shape, ratio_pad, self.input_shape, "Semantic")
        return functional.interpolate(cropped[None, None].float(), size=shape, mode="nearest")[0, 0].to(torch.int64)
