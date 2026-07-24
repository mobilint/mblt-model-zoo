"""Postprocessing for semantic-segmentation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from ..types import ListTensorLike, TensorLike
from ._letterbox import crop_letterbox, get_letterbox_input_shape, resolve_ratio_pads
from .base import PostBase
from .common import RatioPad, normalize_image_shapes


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
        output, is_logits = self._normalize_output(x)
        if img0_shape is None:
            return self._to_input_space(output, is_logits)

        shapes = normalize_image_shapes(img0_shape, output.shape[0])
        pads = resolve_ratio_pads(ratio_pad, output.shape[0], shapes, self.input_shape)
        restored = [
            self._restore(output[index], is_logits, shapes[index], pads[index]) for index in range(output.shape[0])
        ]
        return restored[0] if len(restored) == 1 else restored

    def _normalize_output(self, x: TensorLike | ListTensorLike) -> tuple[torch.Tensor, bool]:
        """Validate one logits tensor or baked class-map tensor."""

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
            return output.to(dtype=torch.float32), True
        if output.ndim == 3:
            class_map = output.to(dtype=torch.int64)
            if class_map.numel() and (int(class_map.min()) < 0 or int(class_map.max()) >= self.nc):
                raise ValueError(f"Semantic class-map values must be in [0, {self.nc - 1}].")
            return class_map, False
        raise ValueError(
            f"Semantic segmentation expects [B, C, H, W] logits or [B, H, W] class maps, got {tuple(output.shape)}."
        )

    def _to_input_space(self, output: torch.Tensor, is_logits: bool) -> torch.Tensor:
        """Restore model output resolution to configured input space."""

        if is_logits:
            if tuple(output.shape[-2:]) != self.input_shape:
                output = functional.interpolate(output, size=self.input_shape, mode="bilinear", align_corners=False)
            return output.argmax(dim=1).to(dtype=torch.int64)
        if tuple(output.shape[-2:]) != self.input_shape:
            output = functional.interpolate(output[:, None].float(), size=self.input_shape, mode="nearest")[:, 0]
        return output.to(dtype=torch.int64)

    def _restore(
        self,
        output: torch.Tensor,
        is_logits: bool,
        shape: tuple[int, int],
        ratio_pad: RatioPad,
    ) -> torch.Tensor:
        """Undo letterboxing, preserving logits until after bilinear restoration."""

        if is_logits:
            if tuple(output.shape[-2:]) != self.input_shape:
                output = functional.interpolate(
                    output[None],
                    size=self.input_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            channels = [crop_letterbox(channel, shape, ratio_pad, self.input_shape, "Semantic") for channel in output]
            cropped_logits = torch.stack(channels)
            restored_logits = functional.interpolate(
                cropped_logits[None],
                size=shape,
                mode="bilinear",
                align_corners=False,
            )[0]
            return restored_logits.argmax(dim=0).to(dtype=torch.int64)

        if tuple(output.shape[-2:]) != self.input_shape:
            output = functional.interpolate(
                output[None, None].float(),
                size=self.input_shape,
                mode="nearest",
            )[0, 0].to(dtype=torch.int64)
        cropped = crop_letterbox(output, shape, ratio_pad, self.input_shape, "Semantic")
        return functional.interpolate(cropped[None, None].float(), size=shape, mode="nearest")[0, 0].to(torch.int64)
