"""Regression tests for ImageNet label utilities."""

from __future__ import annotations

import pytest

from mblt_model_zoo.vision.utils.datasets.imagenet import (
    get_imagenet_label,
    imagenet1000_clsidx_to_labels,
)


@pytest.mark.parametrize(
    ("class_idx", "expected_label"),
    [
        (471, "cannon"),
        (472, "canoe"),
        (518, "crash helmet"),
        (519, "crate"),
    ],
)
def test_imagenet_labels_restore_user_visible_class_names(class_idx: int, expected_label: str) -> None:
    """Keep restored ImageNet labels stable for user-facing results."""

    assert imagenet1000_clsidx_to_labels[class_idx] == expected_label
    assert get_imagenet_label(class_idx) == expected_label
