from mblt_model_zoo.vision import ResNet50
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="/workspace/mblt-model-zoo/tests/rc/volcano.jpg",
    )
    args = parser.parse_args()
    image_path = args.image_path

    resnet50 = ResNet50(
        model_type="IMAGENET1K_V1",
    )

    # resnet50.gpu()
    input_img = resnet50.preprocess(image_path)
    output = resnet50(input_img)
    result = resnet50.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/volcano_resnet50.jpg",
        topk=5,
    )
