from mblt_model_zoo.vision import ResNet50
from argparse import ArgumentParser
from mblt_model_zoo.vision.utils.results import Results

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
        local_model="/workspace/mblt-model-zoo/tmp/resnet50.mxq",
        model_type="IMAGENET1K_V1",
    )
    resnet_pre = resnet50.get_preprocess()
    resnet_post = resnet50.get_postprocess()

    input = resnet_pre(image_path)
    output = resnet50(input)
    result = Results.from_engine(resnet50, resnet_post(output))

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/volcano_resnet50.jpg",
        topk=5,
    )
