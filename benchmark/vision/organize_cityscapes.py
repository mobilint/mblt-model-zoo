"""Organize the Cityscapes validation dataset for local use."""

from __future__ import annotations

import argparse

from mblt_model_zoo.vision.utils.datasets import organize_cityscapes


def main() -> None:
    """Parse organizer options and materialize Cityscapes validation data."""

    parser = argparse.ArgumentParser(description="Organize Cityscapes validation data from Hugging Face")
    parser.add_argument(
        "--output-dir",
        default="~/.mblt_model_zoo/datasets/cityscapes",
        help="Destination for the flat images/ and annotations/ directories",
    )
    args = parser.parse_args()
    organize_cityscapes(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
