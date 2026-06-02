"""
Script to organize the ImageNet dataset.

This script takes local archives or downloadable sources for the ImageNet
dataset and organizes them into a structure suitable for the model zoo.
"""

import argparse
import os

from mblt_model_zoo.vision.utils.datasets import organize_imagenet

DEFAULT_IMAGENET_IMAGE_SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEFAULT_IMAGENET_XML_SOURCE = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"

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
    args = parser.parse_args()

    organize_imagenet(
        image_dir=args.image_dir,
        xml_dir=args.xml_dir,
        output_dir=args.output_dir,
    )
