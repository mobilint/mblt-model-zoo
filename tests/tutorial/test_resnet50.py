import os
from argparse import ArgumentParser
from mblt_model_zoo.vision import ResNet50

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="/workspace/mblt-model-zoo/tests/rc/volcano.jpg",
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
            "/workspace/mblt-model-zoo/tests/tmp/",
            "resnet50_" + os.path.basename(image_path),
        )

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
