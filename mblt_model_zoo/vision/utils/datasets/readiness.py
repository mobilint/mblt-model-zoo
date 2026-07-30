"""Identity and completeness checks for organized vision validation datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..._tasks import normalize_vision_task

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_CLASS_COUNT = 1000
IMAGENET_IMAGES_PER_CLASS = 50
COCO_VALIDATION_SAMPLE_COUNT = 5000
DOTAV1_VALIDATION_SAMPLE_COUNT = 458
WIDERFACE_EVENT_COUNT = 61
WIDERFACE_VALIDATION_SAMPLE_COUNT = 3226
NYU_DEPTH_VALIDATION_SAMPLE_COUNT = 654
ADE20K_VALIDATION_SAMPLE_COUNT = 2000
CITYSCAPES_VALIDATION_SAMPLE_COUNT = 500
IMAGENET_CLASS_PATTERN = re.compile(r"n\d{8}")
IMAGENET_IMAGE_PATTERN = re.compile(r"ILSVRC2012_val_\d{8}")
COCO_IMAGE_PATTERN = re.compile(r"\d{12}")
WIDERFACE_EVENT_PATTERN = re.compile(r"\d+--\S.*")
CITYSCAPES_SAMPLE_ID_PATTERN = re.compile(r"^(?P<city>[A-Za-z][A-Za-z0-9-]*)_\d{6}_\d{6}$")


def _files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path] | None:
    """Collect direct child files with supported suffixes by stem."""

    if not directory.is_dir():
        return {}
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    files = {path.stem: path for path in paths}
    return files if len(files) == len(paths) else None


def _imagenet_ready(root: Path) -> bool:
    """Check the organizer's complete ImageNet-1k validation class tree."""

    if not root.is_dir():
        return False
    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    if len(class_dirs) != IMAGENET_CLASS_COUNT:
        return False
    image_names: set[str] = set()
    for class_dir in class_dirs:
        if IMAGENET_CLASS_PATTERN.fullmatch(class_dir.name) is None:
            return False
        images = [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        if len(images) != IMAGENET_IMAGES_PER_CLASS or any(
            path.suffix != ".JPEG" or IMAGENET_IMAGE_PATTERN.fullmatch(path.stem) is None for path in images
        ):
            return False
        image_names.update(path.name for path in images)
    return len(image_names) == IMAGENET_CLASS_COUNT * IMAGENET_IMAGES_PER_CLASS


def _load_coco_image_names(annotation_path: Path) -> set[str] | None:
    """Load unique validation image filenames from a COCO annotation file."""

    try:
        annotation: Any = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    image_records = annotation.get("images") if isinstance(annotation, dict) else None
    if not isinstance(image_records, list) or len(image_records) != COCO_VALIDATION_SAMPLE_COUNT:
        return None
    names: list[str] = []
    for record in image_records:
        if not isinstance(record, dict):
            continue
        file_name = record.get("file_name")
        if isinstance(file_name, str):
            names.append(file_name)
    return set(names) if len(names) == len(image_records) == len(set(names)) else None


def _coco_ready(root: Path, task: str) -> bool:
    """Check the complete COCO 2017 image split and task annotation metadata."""

    annotation_name = "person_keypoints_val2017.json" if task == "pose_estimation" else "instances_val2017.json"
    annotation_names = _load_coco_image_names(root / annotation_name)
    if annotation_names is None:
        return False
    image_dir = root / "val2017"
    if not image_dir.is_dir():
        return False
    image_paths = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if len(image_paths) != COCO_VALIDATION_SAMPLE_COUNT or any(
        path.suffix.lower() != ".jpg" or COCO_IMAGE_PATTERN.fullmatch(path.stem) is None for path in image_paths
    ):
        return False
    return {path.name for path in image_paths} == annotation_names


def _dotav1_ready(root: Path) -> bool:
    """Check complete paired DOTAv1 validation images and labels."""

    flat_image_dir = root / "images"
    flat_images = _files_by_stem(flat_image_dir, IMAGE_SUFFIXES)
    if flat_images is None:
        return False
    image_dir = flat_image_dir if flat_images else flat_image_dir / "val"
    images = flat_images if flat_images else _files_by_stem(image_dir, IMAGE_SUFFIXES)
    if images is None or len(images) != DOTAV1_VALIDATION_SAMPLE_COUNT:
        return False

    normalized_labels = _files_by_stem(root / "labels" / "val", {".txt"})
    original_labels = _files_by_stem(root / "labels" / "val_original", {".txt"})
    if normalized_labels is None or original_labels is None:
        return False
    label_stems = normalized_labels.keys() | original_labels.keys()
    return images.keys() == label_stems


def _widerface_ready(root: Path) -> bool:
    """Check the complete WiderFace validation image tree and metadata files."""

    required_files = (
        "wider_face_val.mat",
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    )
    if not all((root / file_name).is_file() for file_name in required_files):
        return False
    image_root = root / "images"
    if not image_root.is_dir():
        return False
    event_dirs = [path for path in image_root.iterdir() if path.is_dir()]
    if len(event_dirs) != WIDERFACE_EVENT_COUNT or any(
        WIDERFACE_EVENT_PATTERN.fullmatch(path.name) is None for path in event_dirs
    ):
        return False
    image_count = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for event_dir in event_dirs
        for path in event_dir.iterdir()
    )
    return image_count == WIDERFACE_VALIDATION_SAMPLE_COUNT


def dense_dataset_ready(data_path: str | Path, dataset: str) -> bool:
    """Return whether a dense dataset matches its taxonomy and full validation split.

    Args:
        data_path: Organized dataset root.
        dataset: Dense validation taxonomy.

    Returns:
        Whether the dataset has the expected filename identity, matched targets,
        and complete validation sample count.
    """

    root = Path(data_path).expanduser()
    normalized = dataset.lower()
    if normalized == "nyu-depth":
        images = _files_by_stem(root / "images", {".jpg", ".jpeg", ".png"})
        depths = _files_by_stem(root / "depth", {".npy"})
        if images is None or depths is None:
            return False
        return (
            len(images) == NYU_DEPTH_VALIDATION_SAMPLE_COUNT
            and len(depths) == NYU_DEPTH_VALIDATION_SAMPLE_COUNT
            and images.keys() == depths.keys()
        )

    images = _files_by_stem(root / "images", {".jpg", ".jpeg", ".png"})
    annotations = _files_by_stem(root / "annotations", {".png"})
    if images is None or annotations is None or images.keys() != annotations.keys():
        return False

    if normalized == "ade20k":
        return (
            len(images) == ADE20K_VALIDATION_SAMPLE_COUNT
            and all(stem.startswith("ADE_val_") for stem in images)
            and all(path.suffix.lower() in {".jpg", ".jpeg"} for path in images.values())
        )
    if normalized == "cityscapes":
        return (
            len(images) == CITYSCAPES_VALIDATION_SAMPLE_COUNT
            and all(CITYSCAPES_SAMPLE_ID_PATTERN.fullmatch(stem) is not None for stem in images)
            and all(path.suffix.lower() == ".png" for path in images.values())
        )
    return False


def dataset_ready(data_path: str | Path, task: str, dataset: str | None = None) -> bool:
    """Return whether an organized dataset matches its task, taxonomy, and full validation split.

    Args:
        data_path: Organized dataset root.
        task: Canonical vision task.
        dataset: Optional validation taxonomy.

    Returns:
        Whether the dataset has the expected identity, metadata, and sample count.
    """

    root = Path(data_path).expanduser()
    normalized_task = normalize_vision_task(task)
    expected_dataset = {
        "image_classification": "imagenet",
        "object_detection": "coco",
        "instance_segmentation": "coco",
        "pose_estimation": "coco",
        "face_detection": "widerface",
        "obb": "dotav1",
        "depth_estimation": "nyu-depth",
    }.get(normalized_task)
    normalized_dataset = (dataset or expected_dataset or "").lower()

    if normalized_task == "semantic_segmentation":
        return dense_dataset_ready(root, normalized_dataset or "ade20k")
    if expected_dataset is None or normalized_dataset != expected_dataset:
        return False
    if normalized_task == "image_classification":
        return _imagenet_ready(root)
    if normalized_task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        return _coco_ready(root, normalized_task)
    if normalized_task == "face_detection":
        return _widerface_ready(root)
    if normalized_task == "obb":
        return _dotav1_ready(root)
    if normalized_task == "depth_estimation":
        return dense_dataset_ready(root, normalized_dataset)
    return False
