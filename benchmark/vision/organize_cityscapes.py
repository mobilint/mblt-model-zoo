"""Organize the Cityscapes validation dataset for local use."""

from __future__ import annotations

import argparse

from mblt_model_zoo.vision.utils.datasets import organize_cityscapes


def main() -> None:
    """Parse organizer options and materialize Cityscapes validation data."""

    parser = argparse.ArgumentParser(description="Organize Cityscapes validation data from official ZIP archives")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Path to leftImg8bit_trainvaltest.zip",
    )
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Path to gtFine_trainvaltest.zip",
    )
    parser.add_argument(
        "--output-dir",
        default="~/.mblt_model_zoo/datasets/cityscapes",
        help="Destination for the flat images/ and annotations/ directories",
    )
    args = parser.parse_args()
    organize_cityscapes(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
