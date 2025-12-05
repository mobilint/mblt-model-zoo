import os
import shutil
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory

from tqdm import tqdm


def construct_imagenet(image_dir, xml_dir, output_dir):

    assert len(os.listdir(xml_dir + "/val")) == len(
        os.listdir(image_dir)
    ), f"Number of XML and image files do not match: {len(os.listdir(xml_dir+'/val'))} != {len(os.listdir(image_dir))}"

    # validate the XML files
    pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Validating XML files")
    for xml_file in pbar:
        xml_path = os.path.join(xml_dir + "/val", xml_file)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()

        if len(root.findall("object")) < 1:
            raise ValueError(
                f"XML file {xml_file} has no object, but expected at least 1"
            )

        # check whether the object names in the XML files are the same
        object_names = [obj.find("name").text for obj in root.findall("object")]
        if len(set(object_names)) != 1:
            raise ValueError(
                f"Object names in XML file {xml_file} are not the same. It has {len(set(object_names))} different object names."
            )

    pbar.close()

    # construct the ImageNet dataset
    pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Constructing ImageNet dataset")
    for xml_file in pbar:
        xml_path = os.path.join(xml_dir + "/val", xml_file)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()
        object_name = root.findall("object")[0].find("name").text
        image_path = os.path.join(image_dir, xml_file.replace(".xml", ".JPEG"))
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        os.makedirs(
            os.path.join(output_dir, object_name), exist_ok=True
        )  # create the directory for the object
        shutil.copy(
            image_path,
            os.path.join(output_dir, object_name, os.path.basename(image_path)),
        )  # copy the image to the directory with the same name
    pbar.close()

    # validate the ImageNet dataset
    pbar = tqdm(os.listdir(output_dir), desc="Validating ImageNet dataset")
    print(f"Number of categories: {len(os.listdir(output_dir))}")
    for object_name in pbar:
        num_images = len(os.listdir(os.path.join(output_dir, object_name)))
        if num_images != 50:
            raise ValueError(
                f"Object {object_name} has {num_images} images, but expected 50"
            )
    pbar.close()
    print("Each category has 50 images")
    print("ImageNet dataset constructed successfully")


def organize_imagenet(
    image_dir,
    xml_dir,
    output_dir=os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
):
    if image_dir.endswith(".tar") and xml_dir.endswith(".tgz"):
        with TemporaryDirectory() as temp_dir:
            print("Unpacking image and XML files to temporary directory...")
            shutil.unpack_archive(
                image_dir, os.path.join(temp_dir, "ILSVRC2012_img_val")
            )
            shutil.unpack_archive(
                xml_dir, os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3")
            )
            print("Unpacking completed")
            construct_imagenet(
                os.path.join(temp_dir, "ILSVRC2012_img_val"),
                os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3"),
                output_dir,
            )
    else:
        construct_imagenet(image_dir, xml_dir, output_dir)


def construct_coco(image_dir, annotation_dir, output_dir):
    print(
        f"Constructing COCO dataset from {image_dir} and {annotation_dir} to {output_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        image_dir, os.path.join(output_dir, "val2017"), dirs_exist_ok=True
    )  # copy the image directory to the output directory
    # copy *_val2017.json to the output directory
    for file in os.listdir(os.path.join(annotation_dir, "annotations")):
        if file.endswith("_val2017.json"):
            shutil.copy(
                os.path.join(annotation_dir, "annotations", file),
                os.path.join(output_dir, file),
            )
    print("Constructing COCO dataset completed")


def organize_coco(
    image_dir,
    annotation_dir,
    output_dir=os.path.expanduser("~/.mblt_model_zoo/datasets/coco"),
):
    if image_dir.endswith(".zip") and annotation_dir.endswith(".zip"):
        with TemporaryDirectory() as temp_dir:
            print("Unpacking image and annotation files to temporary directory...")
            shutil.unpack_archive(image_dir, temp_dir)
            shutil.unpack_archive(
                annotation_dir, os.path.join(temp_dir, "annotations_trainval2017")
            )
            print("Unpacking completed")
            construct_coco(
                os.path.join(temp_dir, "val2017"),
                os.path.join(temp_dir, "annotations_trainval2017"),
                output_dir,
            )

    else:
        construct_coco(image_dir, annotation_dir, output_dir)


def construct_widerface(image_dir, annotation_dir, output_dir):
    print(
        f"Constructing WiderFace dataset from {image_dir} and {annotation_dir} to {output_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        os.path.join(image_dir, "images"),
        os.path.join(output_dir, "images"),
        dirs_exist_ok=True,
    )
    for file in os.listdir(annotation_dir):
        if "_val" in file:
            shutil.copy(os.path.join(annotation_dir, file), output_dir)
    print("Constructing WiderFace dataset completed")


def organize_widerface(
    image_dir,
    annotation_dir,
    output_dir=os.path.expanduser("~/.mblt_model_zoo/datasets/widerface"),
):
    if image_dir.endswith(".zip") and annotation_dir.endswith(".zip"):
        with TemporaryDirectory() as temp_dir:
            print("Unpacking image and annotation files to temporary directory...")
            shutil.unpack_archive(image_dir, temp_dir)
            shutil.unpack_archive(annotation_dir, temp_dir)
            print("Unpacking completed")
            construct_widerface(
                os.path.join(temp_dir, "WIDER_val"),
                os.path.join(temp_dir, "wider_face_split"),
                output_dir,
            )
    else:
        construct_widerface(image_dir, annotation_dir, output_dir)


if __name__ == "__main__":
    image_dir = "/workspace/mblt-model-zoo/tests/vision/benchmark/tmp/WIDER_val.zip"
    annotation_dir = (
        "/workspace/mblt-model-zoo/tests/vision/benchmark/tmp/wider_face_split.zip"
    )
    output_dir = "/workspace/mblt-model-zoo/tests/vision/benchmark/tmp/widerface"
    organize_widerface(image_dir, annotation_dir, output_dir)
