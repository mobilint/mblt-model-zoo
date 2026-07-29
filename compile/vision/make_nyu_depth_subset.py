"""Compatibility entry point for NYU Depth calibration subsets."""

from __future__ import annotations

import argparse

from mblt_model_zoo.compile.vision import make_calibration_subset

DEFAULT_DATA_DIR = "~/.mblt_model_zoo/datasets/nyu-depth"


def make_nyu_depth_subset(data_dir: str, output_dir: str, subset_size: int, seed: int = 0) -> None:
    """Create a deterministic NYU Depth calibration subset.

    Args:
        data_dir: Organized NYU Depth root.
        output_dir: Flat subset destination.
        subset_size: Total image count.
        seed: Random selection seed.
    """

    copied = make_calibration_subset("depth_estimation", data_dir, output_dir, subset_size, seed)
    print(f"Created NYU Depth subset with {len(copied)} images at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an NYU Depth calibration subset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the organized NYU Depth dataset")
    parser.add_argument("--output-dir", required=True, help="Path to save the selected images")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of images to select")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to select images")
    args = parser.parse_args()
    make_nyu_depth_subset(args.data_dir, args.output_dir, args.subset_size, args.seed)
