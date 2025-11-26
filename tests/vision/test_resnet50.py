import os
from argparse import ArgumentParser
from pathlib import Path

from mblt_model_zoo.vision import ResNet50

TEST_DIR = Path(__file__).parent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default=os.path.join(TEST_DIR, "rc", "volcano.jpg"),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    image_path = args.image_path
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = os.path.join(
            TEST_DIR, "tmp", f"resnet50_{os.path.basename(image_path)}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    resnet50 = ResNet50()

    # resnet50.gpu()
    input_img = resnet50.preprocess(image_path)
    output = resnet50(input_img)
    result = resnet50.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path=save_path,
        topk=5,
    )
