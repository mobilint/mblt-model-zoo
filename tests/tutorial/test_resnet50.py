from mblt_models.vision import ResNet50
from argparse import ArgumentParser
import PIL
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../rc/volcano.jpg")
    args = parser.parse_args()
    image_path = args.image_path

    resnet50 = ResNet50()
    resnet_pre = resnet50.preprocess
    resnet_post = resnet50.postprocess

    input = resnet_pre(image_path)
    output = resnet50(input)
    result = resnet_post(output)
    print(np.argmax(result))  # expected result: 980 (for volcano)
