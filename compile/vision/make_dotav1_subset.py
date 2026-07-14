"""Create a flat deterministic image subset from an organized DOTAv1 dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_DATA_DIR = "~/.mblt_model_zoo/datasets/dotav1"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def make_dotav1_subset(data_dir: str, output_dir: str, subset_size: int, seed: int) -> None:
    """Copy a deterministic flat DOTAv1 validation-image subset.

    Args:
        data_dir: Root of the organized DOTAv1 dataset.
        output_dir: Destination directory for the selected images.
        subset_size: Number of validation images to select.
        seed: Random seed used to select images.

    Raises:
        ValueError: If the source dataset or requested subset size is invalid.
    """
    source_dir = Path(data_dir).expanduser()
    destination_dir = Path(output_dir).expanduser()
    image_dir = source_dir / "images" / "val"
    image_paths = (
        sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if image_dir.is_dir()
        else []
    )

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("output_dir must be different from data_dir.")
    if not image_paths:
        raise ValueError(f"No DOTAv1 validation images found in {image_dir}.")
    if subset_size <= 0:
        raise ValueError("subset_size must be greater than zero.")
    if subset_size > len(image_paths):
        raise ValueError(f"subset_size ({subset_size}) exceeds the {len(image_paths)} available DOTAv1 images.")

    selected_images = random.Random(seed).sample(image_paths, subset_size)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination_dir.parent, prefix=".dotav1-subset-") as staging_root:
        staging_dir = Path(staging_root)
        for image_path in selected_images:
            shutil.copy2(image_path, staging_dir / image_path.name)

        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.move(str(staging_dir), destination_dir)

    print(f"Created DOTAv1 subset with {subset_size} images at {destination_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a DOTAv1 calibration subset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the organized DOTAv1 dataset")
    parser.add_argument("--output-dir", required=True, help="Path to save the selected images")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of images to select")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select images")
    args = parser.parse_args()

    make_dotav1_subset(args.data_dir, args.output_dir, args.subset_size, args.seed)
