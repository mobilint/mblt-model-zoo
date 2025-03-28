from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../rc/cr7.jpg")
    args = parser.parse_args()
    image_path = args.image_path
