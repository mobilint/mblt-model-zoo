"""Organize the NYU Depth dataset for local use.

The script downloads the published NYU Depth archive by default, or accepts a
local zip file or extracted dataset directory.
"""

import argparse
import os

from mblt_model_zoo.vision.utils.datasets import organize_nyu_depth
from mblt_model_zoo.vision.utils.datasets.organizer import NYU_DEPTH_URL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize NYU Depth dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=NYU_DEPTH_URL,
        help="Local path or download URL for the NYU Depth zip file or extracted dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/nyu-depth"),
        help="Path to the directory to save the organized dataset",
    )
    args = parser.parse_args()

    organize_nyu_depth(dataset_path=args.dataset_path, output_dir=args.output_dir)
