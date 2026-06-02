"""
Script to organize the COCO dataset.

This script takes local archives or downloadable sources for the COCO dataset
and organizes them into a structure suitable for the model zoo.
"""

import argparse
import os

from mblt_model_zoo.vision.utils.datasets import organize_coco

DEFAULT_COCO_IMAGE_SOURCE = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_COCO_ANNOTATION_SOURCE = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

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
    args = parser.parse_args()

    organize_coco(
        image_dir=args.image_dir,
        annotation_dir=args.ann_dir,
        output_dir=args.output_dir,
    )
