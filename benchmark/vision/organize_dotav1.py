"""
Script to organize the DOTAv1 validation dataset.

This script takes a local archive, extracted directory, or downloadable source
for DOTAv1 and organizes only the validation split into a structure suitable for
the model zoo.
"""

import argparse
import os

from mblt_model_zoo.vision.datasets import get_dataset_config
from mblt_model_zoo.vision.utils.datasets import organize_dotav1

DEFAULT_DOTAV1_SOURCE = get_dataset_config("dotav1")["download"]["url"]
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
    args = parser.parse_args()

    organize_dotav1(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )
