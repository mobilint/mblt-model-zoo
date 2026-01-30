import argparse
import os

from mblt_model_zoo.vision.utils.datasets import organize_coco

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize COCO dataset")
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Path to the image zip file"
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        required=True,
        help="Path to the annotation zip file",
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
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
    )
