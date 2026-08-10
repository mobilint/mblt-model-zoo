"""Runtime validation tests for shared Vision preprocessing."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from mblt_vision.utils.preprocess.center_crop import CenterCrop
from mblt_vision.utils.preprocess.letterbox import LetterBox
from mblt_vision.utils.preprocess.normalize import Normalize
from mblt_vision.utils.preprocess.order import SetOrder
from mblt_vision.utils.preprocess.reader import Reader
from mblt_vision.utils.preprocess.resize import Resize


@pytest.mark.parametrize("operation", [CenterCrop(2), Resize(2, "bilinear"), LetterBox([2, 2])])
def test_image_size_operations_reject_nonpositive_sizes(operation: object) -> None:
    """Keep size validation active when Python assertions are optimized away."""

    operation_type = type(operation)
    argument = [0, 2] if operation_type is not Resize else 0
    if operation_type is Resize:
        with pytest.raises(ValueError, match="positive"):
            Resize(argument, "bilinear")
    else:
        with pytest.raises(ValueError, match="positive"):
            operation_type(argument)


@pytest.mark.parametrize("size", [[1], [1, 2, 3], [1.5, 2], "2"])
def test_center_crop_rejects_invalid_size_values(size: object) -> None:
    """Reject invalid crop types and dimensions before processing."""

    with pytest.raises((TypeError, ValueError)):
        CenterCrop(size)  # type: ignore[arg-type]


def test_set_order_rejects_ambiguous_channel_layout() -> None:
    """Do not guess between CHW and HWC when both ends have three channels."""

    with pytest.raises(ValueError, match="ambiguous"):
        SetOrder("HWC")(np.zeros((3, 5, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: Reader("unsupported"), ValueError),
        (lambda: Normalize("unsupported"), ValueError),
        (lambda: SetOrder("unsupported"), ValueError),
    ],
)
def test_preprocessing_styles_raise_explicit_errors(factory, expected: type[Exception]) -> None:
    """Use explicit configuration errors instead of runtime assertions."""

    with pytest.raises(expected):
        factory()


def test_runtime_validation_survives_optimized_python() -> None:
    """Keep configuration checks active under ``python -O``."""

    code = """
from mblt_vision.utils.postprocess import build_postprocess
try:
    build_postprocess(
        {"LetterBox": {"img_size": [640, 640]}},
        {
            "task": "object_detection",
            "dataset": "coco",
            "nl": 3,
            "reg_max": 16,
            "conf_thres": 0,
        },
    )
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True)
