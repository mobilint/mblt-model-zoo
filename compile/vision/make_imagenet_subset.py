"""Create a flat deterministic image subset from an organized ImageNet dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_DATA_DIR = "~/.mblt_model_zoo/datasets/imagenet"


def make_imagenet_subset(data_dir: str, output_dir: str, subset_size: int, seed: int) -> None:
    """Copy a deterministic flat ImageNet subset with a fixed sample count per class.

    Args:
        data_dir: Root of the organized ImageNet dataset.
        output_dir: Destination directory for the selected images.
        subset_size: Number of images to select from every category.
        seed: Random seed used to select images.

    Raises:
        ValueError: If the source dataset or requested subset size is invalid.
    """
    source_dir = Path(data_dir).expanduser()
    destination_dir = Path(output_dir).expanduser()
    category_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir()) if source_dir.is_dir() else []

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("output_dir must be different from data_dir.")
    if not category_dirs:
        raise ValueError(f"No ImageNet category directories found in {source_dir}.")
    if subset_size <= 0:
        raise ValueError("subset_size must be greater than zero.")

    category_images: dict[Path, list[Path]] = {}
    for category_dir in category_dirs:
        images = sorted(path for path in category_dir.iterdir() if path.is_file())
        if subset_size > len(images):
            raise ValueError(
                f"subset_size ({subset_size}) exceeds the {len(images)} available images in {category_dir.name}."
            )
        category_images[category_dir] = images

    random_generator = random.Random(seed)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination_dir.parent, prefix=".imagenet-subset-") as staging_root:
        staging_dir = Path(staging_root)
        for images in category_images.values():
            for image_path in random_generator.sample(images, subset_size):
                shutil.copy2(image_path, staging_dir / image_path.name)

        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.move(str(staging_dir), destination_dir)

    print(
        f"Created ImageNet subset with {subset_size} images from each of {len(category_dirs)} categories "
        f"at {destination_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an ImageNet calibration subset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the organized ImageNet dataset")
    parser.add_argument("--output-dir", required=True, help="Path to save the selected images")
    parser.add_argument("--subset-size", type=int, default=1, help="Number of images to select per category")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select images")
    args = parser.parse_args()

    make_imagenet_subset(args.data_dir, args.output_dir, args.subset_size, args.seed)
