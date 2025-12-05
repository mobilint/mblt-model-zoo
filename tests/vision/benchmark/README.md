# Measuring the Performance of Vision Models

You can run the benchmark as the following steps:

1. Download the vision benchmark dataset and organize the dataset.
2. Prepare the model for benchmark.
3. Run the benchmark and evaluate the performance.

> ⚠️ **Warning:** Mobilint does not provide the vision benchmark dataset or direct download link due to the terms of use of the dataset.

Mobilint model zoo provides utilities on dataset organization, model preparation, and benchmark evaluation, therefore you can easily reproduce the [published benchmark results](../../../mblt_model_zoo/vision/README.md).

Furthermore, you can simply run the benchmark with the model you compiled with custom quantization recipe.

## Benchmark with ImageNet Dataset

### Download the ImageNet Dataset

To download the ImageNet dataset, visit the [ImageNet website](https://image-net.org/) and download the dataset with the following steps:

0. Login with your account. If you don't have an account, you can register a new account. (Using `.edu` email is highly recommended)
1. Click on the "Download" button on the menu bar.
2. Go to "ImageNet Large-scale Visual Recognition Challenge (ILSVRC)" section, and click 2012 button.
3. Download the following files:
    - ILSVRC2012_img_val.tar: Displayes as "Validation images (all tasks)" with MD5 checksum `29b22e2961454d5413ddabcf34fc5622`.
    - ILSVRC2012_bbox_val_v3.tgz: Displayes as "Validation bounding box annotations (all tasks)" with MD5 checksum `f4cd18b5ea29fe6bbea62ec9c20d80f0`.

### Organize the ImageNet Dataset

You can organize the ImageNet dataset with the following command:

```bash
python organize_imagenet.py --image_dir {path_to_ILSVRC2012_img_val.tar} --xml_dir {path_to_ILSVRC2012_bbox_val_v3.tgz} --output_dir ~/.mblt_model_zoo/datasets/imagenet
```

This will organize the dataset into the following structure:

```text
~/.mblt_model_zoo/datasets/imagenet/
├── n01440764/
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ...
├── n01443537/
├── n01484850/
├── ...
```

### Prepare the Model for ImageNet Benchmark

You can prepare the model for ImageNet benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the ImageNet Benchmark

You can run the ImageNetbenchmark with the following command:

```bash
python benchmark_imagenet.py --local_path {path to local mxq(optional)}\
--model_type {model type(optional)}\
--infer_mode {single, multi, global, global4, global8(optional). Default is global}\
--product {aries, regulus(optional). Default is aries}\
--batch_size {batch size(optional). Default is 1}\
--data_path {path to the ImageNet data(optional). Default is ~/.mblt_model_zoo/datasets/imagenet}
```

Example:

```bash
python benchmark_imagenet.py --local_path ./resnet50_IMAGENET1K_V1.mxq --model_type IMAGENET1K_V1 --infer_mode multi --product aries --batch_size 8 --data_path ~/.mblt_model_zoo/datasets/imagenet
```

## Benchmark with COCO Dataset

### Download the COCO Dataset

To download the COCO dataset, visit the [COCO website](https://cocodataset.org/) and download the dataset with the following steps:

1. Click on the "Dataset" button on the menu bar, and click "Download" button.
2. On the "Images" column, click "2017 Val images" button to download `val2017.zip` file.
3. On the "Annotations" column, click "2017 Train/Val annotations" button to download `annotations_trainval2017.zip` file.

### Organize the COCO Dataset

You can organize the COCO dataset with the following command:

```bash
python organize_coco.py --image_dir {path_to_val2017.zip} --annotation_dir {path_to_annotations_trainval2017.zip} --output_dir ~/.mblt_model_zoo/datasets/coco
```

This will organize the dataset into the following structure:

```text
~/.mblt_model_zoo/datasets/coco/
├── val2017/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── ...
├── captions_val2017.json
├── instances_val2017.json
├── person_keypoints_val2017.json
```

### Prepare the Model for COCO Benchmark

You can prepare the model for COCO benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the COCO Benchmark

You can run the COCO benchmark with the following command:


## Benchmark with WiderFace Dataset

### Download the WiderFace Dataset

To download the WiderFace dataset, visit the [WiderFace's Hugging Face page](https://huggingface.co/datasets/CUHK-CSE/wider_face) and download the dataset with the following steps:

1. Go to the "Files and versions" section, and click "data" folder.
2. Download "WIDER_val.zip" file and "wider_face_split.zip" file.

### Organize the WiderFace Dataset

You can organize the WiderFace dataset with the following command:

```bash
python organize_widerface.py --image_dir {path_to_WIDER_val.zip} --annotation_dir {path_to_wider_face_split.zip} --output_dir ~/.mblt_model_zoo/datasets/widerface
```

This will organize the dataset into the following structure:

```text
~/.mblt_model_zoo/datasets/widerface/
├── images/
│   ├── 0--Parade/
|       ├── 0_Parade_Parade_0_102.jpg
|       ├── 0_Parade_Parade_0_120.jpg
|       |...
|   ├── 1--Handshaking/
|       |...
|   ├── ...
├── wider_face_val.mat
├── wider_face_val_bbx_gt.txt
```

### Prepare the Model for WiderFace Benchmark

You can prepare the model for WiderFace benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the WiderFace Benchmark

You can run the WiderFace benchmark with the following command:
