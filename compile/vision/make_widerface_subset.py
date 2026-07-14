"""Compatibility entry point for WiderFace calibration subsets."""

from __future__ import annotations

import argparse

from mblt_model_zoo.compile.vision import make_calibration_subset

DEFAULT_DATA_DIR = "~/.mblt_model_zoo/datasets/widerface"


def make_widerface_subset(data_dir: str, output_dir: str, subset_size: int, seed: int) -> None:
    """Create a deterministic WiderFace calibration subset.

    Args:
        data_dir: Organized WiderFace root.
        output_dir: Flat subset destination.
        subset_size: Images selected per category.
        seed: Random selection seed.
    """

    copied = make_calibration_subset("face_detection", data_dir, output_dir, subset_size, seed)
    print(f"Created WiderFace subset with {len(copied)} images at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a WiderFace calibration subset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the organized WiderFace dataset")
    parser.add_argument("--output-dir", required=True, help="Path to save the selected images")
    parser.add_argument("--subset-size", type=int, default=1, help="Number of images to select per category")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select images")
    args = parser.parse_args()
    make_widerface_subset(args.data_dir, args.output_dir, args.subset_size, args.seed)
