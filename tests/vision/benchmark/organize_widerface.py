import argparse
import os

from mblt_model_zoo.vision import organize_widerface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize WiderFace dataset")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the image zip file"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="Path to the annotation zip file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/widerface"),
        help="Path to the directory to save the organized dataset",
    )
    args = parser.parse_args()

    organize_widerface(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
    )
