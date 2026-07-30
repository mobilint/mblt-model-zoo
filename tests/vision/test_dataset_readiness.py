"""Tests for organized vision dataset identity and completeness checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from mblt_model_zoo.vision.utils.datasets import readiness


def _write_file(path: Path) -> None:
    """Create a placeholder dataset file and its parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def _write_widerface_metadata(path: Path, event_images: dict[str, list[str]]) -> None:
    """Write the WiderFace event and image-name cell arrays used by readiness."""

    event_list = np.empty((len(event_images), 1), dtype=object)
    file_list = np.empty((len(event_images), 1), dtype=object)
    for index, (event_name, image_stems) in enumerate(event_images.items()):
        event_list[index, 0] = event_name
        file_list[index, 0] = np.array([[stem] for stem in image_stems], dtype=object)
    savemat(path, {"event_list": event_list, "file_list": file_list})


def test_imagenet_readiness_requires_complete_official_class_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject incomplete or non-ImageNet classification directory trees."""

    monkeypatch.setattr(readiness, "IMAGENET_CLASS_COUNT", 2)
    monkeypatch.setattr(readiness, "IMAGENET_IMAGES_PER_CLASS", 2)
    for class_index in range(2):
        for image_index in range(2):
            if (class_index, image_index) == (1, 1):
                continue
            _write_file(
                tmp_path / f"n{class_index:08d}" / f"ILSVRC2012_val_{class_index * 2 + image_index + 1:08d}.JPEG"
            )

    assert not readiness.dataset_ready(tmp_path, "image_classification", "imagenet")

    _write_file(tmp_path / "n00000001" / "ILSVRC2012_val_00000004.JPEG")

    assert readiness.dataset_ready(tmp_path, "image_classification", "imagenet")
    assert not readiness.dataset_ready(tmp_path, "image_classification", "coco")


@pytest.mark.parametrize(
    ("task", "annotation_name"),
    [
        ("object_detection", "instances_val2017.json"),
        ("instance_segmentation", "instances_val2017.json"),
        ("pose_estimation", "person_keypoints_val2017.json"),
    ],
)
def test_coco_readiness_matches_images_to_task_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    annotation_name: str,
) -> None:
    """Require every official COCO image in the task-specific annotation file."""

    monkeypatch.setattr(readiness, "COCO_VALIDATION_SAMPLE_COUNT", 2)
    image_names = ["000000000001.jpg", "000000000002.jpg"]
    for image_name in image_names:
        _write_file(tmp_path / "val2017" / image_name)
    (tmp_path / annotation_name).write_text(
        json.dumps({"images": [{"file_name": image_names[0]}]}),
        encoding="utf-8",
    )

    assert not readiness.dataset_ready(tmp_path, task, "coco")

    (tmp_path / annotation_name).write_text(
        json.dumps({"images": [{"file_name": image_name} for image_name in image_names]}),
        encoding="utf-8",
    )

    assert readiness.dataset_ready(tmp_path, task, "coco")
    assert not readiness.dataset_ready(tmp_path, task, "imagenet")


@pytest.mark.parametrize("relative_image_dir", ["images", "images/val"])
def test_dotav1_readiness_requires_complete_image_label_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_image_dir: str,
) -> None:
    """Accept flat and legacy DOTA images only when every image has a label."""

    monkeypatch.setattr(readiness, "DOTAV1_VALIDATION_SAMPLE_COUNT", 2)
    for stem in ("P0001", "P0002"):
        _write_file(tmp_path / relative_image_dir / f"{stem}.png")
    _write_file(tmp_path / "labels" / "val_original" / "P0001.txt")

    assert not readiness.dataset_ready(tmp_path, "obb", "dotav1")

    _write_file(tmp_path / "labels" / "val" / "P0002.txt")

    assert readiness.dataset_ready(tmp_path, "obb", "dotav1")


def test_widerface_readiness_requires_complete_event_tree_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require all WiderFace validation events, images, and evaluation metadata."""

    monkeypatch.setattr(readiness, "WIDERFACE_EVENT_COUNT", 2)
    monkeypatch.setattr(readiness, "WIDERFACE_VALIDATION_SAMPLE_COUNT", 2)
    _write_widerface_metadata(
        tmp_path / "wider_face_val.mat",
        {"0--Parade": ["sample-0"], "1--Handshaking": ["sample-1"]},
    )
    for file_name in ("wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"):
        _write_file(tmp_path / file_name)
    _write_file(tmp_path / "images" / "0--Parade" / "sample-0.jpg")
    (tmp_path / "images" / "1--Handshaking").mkdir()

    assert not readiness.dataset_ready(tmp_path, "face_detection", "widerface")

    _write_file(tmp_path / "images" / "1--Handshaking" / "sample-1.jpg")

    assert readiness.dataset_ready(tmp_path, "face_detection", "widerface")
    assert not readiness.dataset_ready(tmp_path, "face_detection", "coco")


@pytest.mark.parametrize(
    ("event_name", "image_name"),
    [
        ("0--Parade", "stale"),
        ("1--Handshaking", "expected"),
    ],
)
def test_widerface_readiness_rejects_tree_not_named_by_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    event_name: str,
    image_name: str,
) -> None:
    """Reject complete-looking trees whose event or image identity differs from metadata."""

    monkeypatch.setattr(readiness, "WIDERFACE_EVENT_COUNT", 1)
    monkeypatch.setattr(readiness, "WIDERFACE_VALIDATION_SAMPLE_COUNT", 1)
    _write_widerface_metadata(tmp_path / "wider_face_val.mat", {"0--Parade": ["expected"]})
    for file_name in ("wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"):
        _write_file(tmp_path / file_name)
    _write_file(tmp_path / "images" / event_name / f"{image_name}.jpg")

    assert not readiness.dataset_ready(tmp_path, "face_detection", "widerface")
