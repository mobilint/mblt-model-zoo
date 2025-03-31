from mblt_models.vision import ResNet50
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../rc/volcano.jpg")
    args = parser.parse_args()
    image_path = args.image_path

    resnet50 = ResNet50(
        local_model="/workspace/mblt-models/tmp/resnet50.mxq",
        model_type="IMAGENET1K_V1",
    )
    resnet_pre = resnet50.get_preprocess()
    resnet_post = resnet50.get_postprocess()

    input = resnet_pre(image_path)
    output = resnet50(input)
    result = resnet_post(output)
    print(np.argmax(result))  # expected result: 980 (for volcano)
