"""Identity and completeness checks for organized dense validation datasets."""

from __future__ import annotations

import re
from pathlib import Path

NYU_DEPTH_VALIDATION_SAMPLE_COUNT = 654
ADE20K_VALIDATION_SAMPLE_COUNT = 2000
CITYSCAPES_VALIDATION_SAMPLE_COUNT = 500
CITYSCAPES_SAMPLE_ID_PATTERN = re.compile(r"^(?P<city>[A-Za-z][A-Za-z0-9-]*)_\d{6}_\d{6}$")


def _files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path] | None:
    """Collect direct child files with supported suffixes by stem."""

    if not directory.is_dir():
        return {}
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    files = {path.stem: path for path in paths}
    return files if len(files) == len(paths) else None


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
