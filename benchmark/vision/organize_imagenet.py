import argparse
import os

from mblt_model_zoo.vision.utils.datasets import organize_imagenet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize ImageNet dataset")
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Path to the image tar file"
    )
    parser.add_argument(
        "--xml-dir", type=str, required=True, help="Path to the xml tgz file"
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
