"""Compatibility entry point for COCO calibration subsets."""

from __future__ import annotations

import argparse

from mblt_model_zoo.compile.vision import make_calibration_subset

DEFAULT_DATA_DIR = "~/.mblt_model_zoo/datasets/coco"


def make_coco_subset(data_dir: str, output_dir: str, subset_size: int, seed: int) -> None:
    """Create a deterministic COCO calibration subset.

    Args:
        data_dir: Organized COCO root.
        output_dir: Flat subset destination.
        subset_size: Total image count.
        seed: Random selection seed.
    """

    copied = make_calibration_subset("object_detection", data_dir, output_dir, subset_size, seed)
    print(f"Created COCO subset with {len(copied)} images at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a COCO calibration subset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the organized COCO dataset")
    parser.add_argument("--output-dir", required=True, help="Path to save the selected images")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of images to select")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select images")
    args = parser.parse_args()
    make_coco_subset(args.data_dir, args.output_dir, args.subset_size, args.seed)
