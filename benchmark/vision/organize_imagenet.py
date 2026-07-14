"""
Script to organize the ImageNet dataset.

This script takes local archives or downloadable sources for the ImageNet
dataset and organizes them into a structure suitable for the model zoo.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from mblt_model_zoo.vision.utils.datasets import organize_imagenet

DEFAULT_IMAGENET_IMAGE_SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEFAULT_IMAGENET_XML_SOURCE = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
IMAGENET_CLASS_COUNT = 1000
IMAGENET_IMAGES_PER_CLASS = 50


def _has_organized_imagenet_dataset(output_dir: Path) -> bool:
    """Returns whether a complete organized ImageNet validation dataset is available.

    Args:
        output_dir: Root of the organized ImageNet dataset.

    Returns:
        Whether the directory contains all 1,000 validation classes with 50 images each.
    """
    category_dirs = [path for path in output_dir.iterdir() if path.is_dir()] if output_dir.is_dir() else []
    return len(category_dirs) == IMAGENET_CLASS_COUNT and all(
        sum(path.is_file() for path in category_dir.iterdir()) == IMAGENET_IMAGES_PER_CLASS
        for category_dir in category_dirs
    )


def make_imagenet_subset(output_dir: str, subset_dir: str, subset_size: int, seed: int) -> None:
    """Creates a flat deterministic ImageNet subset with a fixed sample count per class.

    Args:
        output_dir: Root of the organized ImageNet dataset.
        subset_dir: Destination directory for the subset.
        subset_size: Number of images to select from every category.
        seed: Random seed used to select images.

    Raises:
        ValueError: If the requested subset is invalid or a category has too few images.
    """
    source_dir = Path(output_dir).expanduser()
    destination_dir = Path(subset_dir).expanduser()
    category_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir()) if source_dir.is_dir() else []

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("subset_dir must be different from output_dir.")
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
    destination_parent = destination_dir.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination_parent, prefix=".imagenet-subset-") as staging_root:
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
    parser = argparse.ArgumentParser(description="Organize ImageNet dataset")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGENET_IMAGE_SOURCE,
        help="Local path or download URL for the image tar file",
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default=DEFAULT_IMAGENET_XML_SOURCE,
        help="Local path or download URL for the XML tgz file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
        help="Path to the directory to save the organized dataset",
    )
    parser.add_argument("--subset-dir", type=str, help="Path to save a deterministic validation subset")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1,
        help="Number of validation images to select from each category",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select subset images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if args.subset_dir is None or not _has_organized_imagenet_dataset(output_dir):
        organize_imagenet(
            image_dir=args.image_dir,
            xml_dir=args.xml_dir,
            output_dir=str(output_dir),
        )
    else:
        print(f"Using existing ImageNet dataset at {output_dir}")

    if args.subset_dir is not None:
        make_imagenet_subset(str(output_dir), args.subset_dir, args.subset_size, args.seed)
