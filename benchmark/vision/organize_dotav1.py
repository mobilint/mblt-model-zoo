"""
Script to organize the DOTAv1 validation dataset.

This script takes a local archive, extracted directory, or downloadable source
for DOTAv1 and organizes only the validation split into a structure suitable for
the model zoo.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from mblt_model_zoo.vision.datasets import get_dataset_config
from mblt_model_zoo.vision.utils.datasets import organize_dotav1

DEFAULT_DOTAV1_SOURCE = get_dataset_config("dotav1")["download"]["url"]
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _has_organized_dotav1_dataset(output_dir: Path) -> bool:
    """Returns whether an organized DOTAv1 validation dataset is available.

    Args:
        output_dir: Root of the organized DOTAv1 dataset.

    Returns:
        Whether the validation images and original labels are present.
    """
    return (output_dir / "images" / "val").is_dir() and (output_dir / "labels" / "val_original").is_dir()


def make_dotav1_subset(output_dir: str, subset_dir: str, subset_size: int, seed: int) -> None:
    """Creates a flat deterministic DOTAv1 validation-image subset.

    Args:
        output_dir: Root of the organized DOTAv1 dataset.
        subset_dir: Destination directory for the subset.
        subset_size: Number of validation images to include.
        seed: Random seed used to select images.

    Raises:
        ValueError: If the requested subset is invalid or the dataset is incomplete.
    """
    source_dir = Path(output_dir).expanduser()
    destination_dir = Path(subset_dir).expanduser()
    image_dir = source_dir / "images" / "val"
    image_paths = (
        sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if image_dir.is_dir()
        else []
    )

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("subset_dir must be different from output_dir.")
    if not _has_organized_dotav1_dataset(source_dir):
        raise ValueError(f"DOTAv1 dataset is incomplete in {source_dir}.")
    if subset_size <= 0:
        raise ValueError("subset_size must be greater than zero.")
    if subset_size > len(image_paths):
        raise ValueError(f"subset_size ({subset_size}) exceeds the {len(image_paths)} available DOTAv1 images.")

    selected_images = random.Random(seed).sample(image_paths, subset_size)
    destination_parent = destination_dir.parent
    destination_parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination_parent, prefix=".dotav1-subset-") as staging_root:
        staging_dir = Path(staging_root)
        for image_path in selected_images:
            shutil.copy2(image_path, staging_dir / image_path.name)

        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.move(str(staging_dir), destination_dir)

    print(f"Created DOTAv1 subset with {subset_size} images at {destination_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize DOTAv1 validation dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DOTAV1_SOURCE,
        help="Local path, archive URL, or Google Drive folder URL for DOTAv1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/dotav1"),
        help="Path to the directory to save the organized dataset",
    )
    parser.add_argument("--subset-dir", type=str, help="Path to save a deterministic validation subset")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of validation images in the subset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select subset images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if args.subset_dir is None or not _has_organized_dotav1_dataset(output_dir):
        organize_dotav1(
            dataset_path=args.dataset_path,
            output_dir=str(output_dir),
        )
    else:
        print(f"Using existing DOTAv1 dataset at {output_dir}")

    if args.subset_dir is not None:
        make_dotav1_subset(str(output_dir), args.subset_dir, args.subset_size, args.seed)
