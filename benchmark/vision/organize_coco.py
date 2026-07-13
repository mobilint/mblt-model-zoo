"""
Script to organize the COCO dataset.

This script takes local archives or downloadable sources for the COCO dataset
and organizes them into a structure suitable for the model zoo.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from mblt_model_zoo.vision.utils.datasets import organize_coco

DEFAULT_COCO_IMAGE_SOURCE = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_COCO_ANNOTATION_SOURCE = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _has_organized_coco_dataset(output_dir: Path) -> bool:
    """Returns whether an organized COCO validation dataset is available.

    Args:
        output_dir: Root of the organized COCO dataset.

    Returns:
        Whether the validation images and instance annotations are present.
    """
    return (output_dir / "val2017").is_dir() and (output_dir / "instances_val2017.json").is_file()


def make_coco_subset(output_dir: str, subset_dir: str, subset_size: int, seed: int) -> None:
    """Creates a deterministic COCO validation subset with matching annotations.

    Args:
        output_dir: Root of the organized COCO dataset.
        subset_dir: Destination directory for the subset.
        subset_size: Number of validation images to include.
        seed: Random seed used to select images.

    Raises:
        ValueError: If the requested subset is invalid or the dataset is incomplete.
    """
    source_dir = Path(output_dir).expanduser()
    destination_dir = Path(subset_dir).expanduser()
    image_dir = source_dir / "val2017"
    annotation_paths = sorted(source_dir.glob("*_val2017.json"))
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else []

    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("subset_dir must be different from output_dir.")
    if not _has_organized_coco_dataset(source_dir):
        raise ValueError(f"COCO dataset is incomplete in {source_dir}.")
    if subset_size <= 0:
        raise ValueError("subset_size must be greater than zero.")
    if subset_size > len(image_paths):
        raise ValueError(f"subset_size ({subset_size}) exceeds the {len(image_paths)} available COCO images.")

    selected_images = random.Random(seed).sample(image_paths, subset_size)
    selected_names = {image_path.name for image_path in selected_images}
    destination_parent = destination_dir.parent
    destination_parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination_parent, prefix=".coco-subset-") as staging_root:
        staging_dir = Path(staging_root)
        staging_image_dir = staging_dir / "val2017"
        staging_image_dir.mkdir()
        for image_path in selected_images:
            shutil.copy2(image_path, staging_image_dir / image_path.name)

        for annotation_path in annotation_paths:
            with annotation_path.open(encoding="utf-8") as file:
                annotation_data = json.load(file)
            selected_ids = {
                image["id"] for image in annotation_data.get("images", []) if image.get("file_name") in selected_names
            }
            annotation_data["images"] = [
                image for image in annotation_data.get("images", []) if image.get("id") in selected_ids
            ]
            annotation_data["annotations"] = [
                annotation
                for annotation in annotation_data.get("annotations", [])
                if annotation.get("image_id") in selected_ids
            ]
            with (staging_dir / annotation_path.name).open("w", encoding="utf-8") as file:
                json.dump(annotation_data, file)

        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.move(str(staging_dir), destination_dir)

    print(f"Created COCO subset with {subset_size} images at {destination_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize COCO dataset")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_COCO_IMAGE_SOURCE,
        help="Local path or download URL for the image zip file",
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default=DEFAULT_COCO_ANNOTATION_SOURCE,
        help="Local path or download URL for the annotation zip file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/coco"),
        help="Path to the directory to save the organized dataset",
    )
    parser.add_argument("--subset-dir", type=str, help="Path to save a deterministic validation subset")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of validation images in the subset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select subset images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if args.subset_dir is None or not _has_organized_coco_dataset(output_dir):
        organize_coco(
            image_dir=args.image_dir,
            annotation_dir=args.ann_dir,
            output_dir=str(output_dir),
        )
    else:
        print(f"Using existing COCO dataset at {output_dir}")

    if args.subset_dir is not None:
        make_coco_subset(str(output_dir), args.subset_dir, args.subset_size, args.seed)
