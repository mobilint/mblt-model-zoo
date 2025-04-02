from typing import List, Union, Tuple
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from .base import PreOps
from ..types import TensorLike

PIL_INTERP_CODES = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "box": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING,
    "lanczos": Image.Resampling.LANCZOS,
}


class Resize(PreOps):
    """Resize image in Torch backend"""

    def __init__(
        self,
        size: Union[int, List[int]],
        interpolation: str,
    ):
        # Note that this behaves different for npy image and PIL image
        super().__init__()
        self.size = size  # h, w
        self.interpolation = interpolation

    def __call__(self, x: Union[TensorLike, Image.Image]):
        """Resize image

        Args:
            x (Union[np.ndarray, Image]): Image to be resized.

        Raises:
            TypeError: If x is not numpy array or PIL image.

        Returns:
            x: Resized image.
        """
        # result: np.ndarray (H, W, C)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if isinstance(x, torch.Tensor):
            assert x.ndim() == 3, f"Got unexpected x.shape={x.shape}."
            x = x.to(self.device)
            img_h, img_w = x.shape[:2]
            new_h, new_w = self._compute_resized_output_size(img_h, img_w)

            if [img_h, img_w] == [new_h, new_w]:
                return x

            x, need_cast, need_squeeze, out_dtype = self._cast_squeeze_in(
                x, [torch.float32, torch.float64]
            )

            x = F.interpolate(
                x[None],
                size=(new_h, new_w),
                mode=self.interpolation,
                align_corners=(
                    False if self.interpolation in ["bilinear", "bicubic"] else None
                ),
                antialias=self.interpolation in ["bilinear", "bicubic"],
            )

            x = self._cast_squeeze_out(x, need_cast, need_squeeze, out_dtype)

            return x.to(self.device)
        elif isinstance(x, Image.Image):
            img_w, img_h = x.size
            new_h, new_w = self._compute_resized_output_size(img_h, img_w)
            if [img_h, img_w] == [new_h, new_w]:
                return x
            return x.resize(
                size=(new_w, new_h),
                resample=PIL_INTERP_CODES[self.interpolation],
            )
        else:
            raise TypeError(f"Got unexpected type for x={type(x)}.")

    def _compute_resized_output_size(self, img_h: int, img_w: int):
        if isinstance(self.size, int):
            # to match the shortest side to self.size with the same ratio
            ratio = max(self.size / img_h, self.size / img_w)
            new_h = int(round(img_h * ratio))
            new_w = int(round(img_w * ratio))
        elif isinstance(self.size, list) and len(self.size) == 2:
            new_h, new_w = self.size
        else:
            raise ValueError(f"Got unexpected size={self.size}.")

        return [new_h, new_w]

    def _cast_squeeze_in(
        self, img: torch.Tensor, req_dtypes: List[torch.dtype]
    ) -> Tuple[torch.Tensor, bool, bool, torch.dtype]:
        need_squeeze = False
        # make image NCHW
        if img.ndim < 4:
            img = img.unsqueeze(dim=0)
            need_squeeze = True

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)
        return img, need_cast, need_squeeze, out_dtype

    def _cast_squeeze_out(
        self,
        img: torch.Tensor,
        need_cast: bool,
        need_squeeze: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if need_squeeze:
            img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                # it is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img
