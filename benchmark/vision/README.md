# Measuring the Performance of Vision Models

You can run the benchmark as the following steps:

1. Download the vision benchmark dataset and organize the dataset.
2. Prepare the model for benchmark.
3. Run the benchmark and evaluate the performance.

> ⚠️ **Warning:** Mobilint does not host the vision benchmark datasets.
> The organizer scripts use the datasets' official downloadable sources by default and can also
> work with local archives that you downloaded yourself.
> Mobilint model zoo provides utilities on dataset organization, model preparation, and benchmark
> evaluation, therefore you can easily reproduce the
> [published benchmark results](../../mblt_model_zoo/vision/README.md).

Furthermore, you can simply run the benchmark with the model you compiled with custom quantization recipe.

## Benchmark with ImageNet Dataset

### Download the ImageNet Dataset

The organizer uses ImageNet's official validation image and annotation archives by default. If you
already downloaded those files yourself, you can point the script to the local archive paths
instead.

### Organize the ImageNet Dataset

You can organize the ImageNet dataset with the following command:

```bash
python benchmark/vision/organize_imagenet.py \
  --output-dir ~/.mblt_model_zoo/datasets/imagenet
```

If you want to use local archives instead of the built-in download sources:

```bash
python benchmark/vision/organize_imagenet.py \
  --image-dir {path_to_ILSVRC2012_img_val.tar} \
  --xml-dir {path_to_ILSVRC2012_bbox_val_v3.tgz} \
  --output-dir ~/.mblt_model_zoo/datasets/imagenet
```

To create a reproducible calibration subset, add `--subset-dir`, an optional per-category
`--subset-size`, and an optional `--seed`. The organizer reuses a complete `--output-dir` when it
is available; otherwise it organizes the supplied archives (or downloads the defaults) first:

```bash
python benchmark/vision/organize_imagenet.py \
  --image-dir {path_to_ILSVRC2012_img_val.tar} \
  --xml-dir {path_to_ILSVRC2012_bbox_val_v3.tgz} \
  --output-dir ~/.mblt_model_zoo/datasets/imagenet \
  --subset-dir /workspace/calib_imagenet \
  --subset-size 1 \
  --seed 0
```

`--subset-size` defaults to 1. ImageNet has 1,000 validation categories, so the default subset
contains 1,000 images total. Reusing the same seed produces the same selection.

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

You can prepare the model for ImageNet benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the ImageNet Benchmark

You can run the ImageNet benchmark with the following command:

```bash
python benchmark/vision/benchmark_imagenet.py \
  --model-cls {model class(optional). Default is resnet50} \
  --mxq-path {path to local mxq(optional)} \
  --model-type {model type(optional). Default is DEFAULT} \
  --core-mode {single, multi, global4, global8(optional). Default is global8} \
  --batch-size {batch size(optional). Default is 1} \
  --data-path {path to the ImageNet data(optional). Default is ~/.mblt_model_zoo/datasets/imagenet}
```

Example:

```bash
python benchmark/vision/benchmark_imagenet.py \
  --model-cls resnet50 \
  --mxq-path ./resnet50_IMAGENET1K_V1.mxq \
  --model-type IMAGENET1K_V1 \
  --core-mode multi \
  --batch-size 8 \
  --data-path ~/.mblt_model_zoo/datasets/imagenet
```

## Benchmark with COCO Dataset

### Download the COCO Dataset

The organizer uses COCO's official validation image and annotation archives by default. If you
already downloaded those files yourself, you can pass the local archive paths instead.

### Organize the COCO Dataset

You can organize the COCO dataset with the following command:

```bash
python benchmark/vision/organize_coco.py \
  --output-dir ~/.mblt_model_zoo/datasets/coco
```

To use local archives:

```bash
python benchmark/vision/organize_coco.py \
  --image-dir {path_to_val2017.zip} \
  --ann-dir {path_to_annotations_trainval2017.zip} \
  --output-dir ~/.mblt_model_zoo/datasets/coco
```

To create a reproducible calibration subset, the organizer first reuses the complete dataset in
`--output-dir` when it is available. Otherwise, it organizes the supplied archives (or downloads
the defaults) before creating the subset:

```bash
python benchmark/vision/organize_coco.py \
  --image-dir {path_to_val2017.zip} \
  --ann-dir {path_to_annotations_trainval2017.zip} \
  --output-dir ~/.mblt_model_zoo/datasets/coco \
  --subset-dir /workspace/calib_coco \
  --subset-size 100 \
  --seed 0
```

The subset includes the selected images and filtered `*_val2017.json` annotations. `--subset-size`
defaults to 100, and reusing the same seed produces the same selection.

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

You can prepare the model for COCO benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the COCO Benchmark

You can run the COCO benchmark with the following command:

```bash
python benchmark/vision/benchmark_coco.py \
  --model-cls {model class(optional). Default is YOLOv5m} \
  --mxq-path {path to local mxq(optional)} \
  --model-type {model type(optional). Default is DEFAULT} \
  --core-mode {single, multi, global4, global8(optional). Default is global8} \
  --batch-size {batch size(optional). Default is 1} \
  --data-path {path to the COCO data(optional). Default is ~/.mblt_model_zoo/datasets/coco} \
  --conf-thres {confidence threshold for object detection(optional). Default is 0.001} \
  --iou-thres {IOU threshold for object detection(optional). Default is 0.7}
```

Example:

```bash
python benchmark/vision/benchmark_coco.py \
  --model-cls YOLOv5m \
  --core-mode single \
  --batch-size 8 \
  --data-path ~/.mblt_model_zoo/datasets/coco \
  --conf-thres 0.001 \
  --iou-thres 0.7
```

## Benchmark with WiderFace Dataset

### Download the WiderFace Dataset

The organizer uses the official WiderFace validation image and split archives by default. If you
already downloaded those files yourself, you can pass the local archive paths instead.

### Organize the WiderFace Dataset

You can organize the WiderFace dataset with the following command:

```bash
python benchmark/vision/organize_widerface.py \
  --output-dir ~/.mblt_model_zoo/datasets/widerface
```

To use local archives:

```bash
python benchmark/vision/organize_widerface.py \
  --image-dir {path_to_WIDER_val.zip} \
  --annotation-dir {path_to_wider_face_split.zip} \
  --output-dir ~/.mblt_model_zoo/datasets/widerface
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

You can prepare the model for WiderFace benchmark or simply run the benchmark with the model that is provided in [mblt-model-zoo](../../mblt_model_zoo/vision/README.md).

If you want to try with your own model, refer to the [tutorial guide](https://github.com/mobilint/mblt-sdk-tutorial/) to compile the model with custom quantization recipe.

### Run the WiderFace Benchmark

Pending

## Benchmark with DOTAv1 Dataset

### Download the DOTAv1 Dataset

The organizer reads the default source from `mblt_model_zoo/vision/datasets/dotav1.yaml` and
downloads the DOTAv1 validation images and original v1.0 labels from its Google Drive folder. If
you already downloaded the archive or extracted it locally, you can pass that local path instead.

### Organize the DOTAv1 Dataset

You can organize the DOTAv1 validation dataset with the following command:

```bash
python benchmark/vision/organize_dotav1.py \
  --output-dir ~/.mblt_model_zoo/datasets/dotav1
```

To use a local archive or extracted dataset directory:

```bash
python benchmark/vision/organize_dotav1.py \
  --dataset-path {path_to_DOTAv1.zip_or_directory} \
  --output-dir ~/.mblt_model_zoo/datasets/dotav1
```

To create a reproducible calibration subset, add `--subset-dir`, an optional `--subset-size`, and
an optional `--seed`. The organizer reuses a complete `--output-dir` when possible; otherwise it
first organizes the supplied source or downloads the default archives:

```bash
python benchmark/vision/organize_dotav1.py \
  --dataset-path {path_to_DOTAv1.zip_or_directory} \
  --output-dir ~/.mblt_model_zoo/datasets/dotav1 \
  --subset-dir /workspace/calib_dotav1 \
  --subset-size 100 \
  --seed 0
```

The subset includes each selected validation image and its original label, plus converted labels
when they already exist in the full dataset. `--subset-size` defaults to 100.

This keeps only the validation split and organizes the dataset into the following structure:

```text
~/.mblt_model_zoo/datasets/dotav1/
├── images/
│   └── val/
│       ├── P0003.png
│       ├── P0006.png
│       ├── ...
├── labels/
│   ├── val/
│   │   ├── P0003.txt
│   │   ├── P0006.txt
│   │   ├── ...
│   └── val_original/
│       ├── P0003.txt
│       ├── P0006.txt
│       ├── ...
```

### Run the DOTAv1 Benchmark

You can run the DOTAv1 benchmark with the following command:

```bash
python benchmark/vision/benchmark_dota.py \
  --model-cls YOLOv8s-obb \
  --framework onnx \
  --batch-size 1 \
  --data-path ~/.mblt_model_zoo/datasets/dotav1 \
  --conf-thres 0.01 \
  --iou-thres 0.7
```

The benchmark reports local rotated `mAP test 50` and `mAP test 50-95`, and writes DOTA Task1
prediction text files under `benchmark/vision/results/dota`.

## Compare Vision Benchmark Results

You can compare multiple vision benchmark CSV files and generate model-wise charts:

```bash
python benchmark/vision/plot_compare_benchmark_results.py \
  ./results/results_a4000.csv \
  ./results/results_a5000.csv \
  ./results/results_mla100.csv
```

Output charts are saved under:

```text
benchmark/vision/results/charts/<input1_input2_...>/
```

You can also pass directories instead of explicit CSV files when each directory contains a single benchmark CSV
or a `results.csv` file.
