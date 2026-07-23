# Vision Benchmark Guide

The vision benchmark follows the same artifact-oriented workflow as
[`benchmark/transformers/`](../transformers/README.md): run one or more targets, then keep the
generated JSON, CSV, Markdown summary, and chart together in a results directory. The legacy
task-specific scripts have been replaced by the standard runner below.

## Quick CLI Validation

Use `mblt-model-zoo val` when you want to validate one vision model without manually selecting a
dataset evaluator. The CLI infers the model task, selects ImageNet, COCO, WiderFace, or DOTAv1 as
appropriate, and organizes the default dataset cache if it is missing.

```bash
mblt-model-zoo val --help

mblt-model-zoo val \
  --model resnet50 \
  --model-path ./resnet50.mxq \
  --core-mode global8 \
  --data-path ~/.mblt_model_zoo/datasets/imagenet

mblt-model-zoo val \
  --model yolo11m \
  --batch-size 8 \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --data-path ~/.mblt_model_zoo/datasets/coco
```

Image classification validation uses Top-1 accuracy as its primary score and Top-5 accuracy as its
secondary metric. Other tasks report their task-specific score, such as COCO mAP, WiderFace AP, or
DOTAv1 rotated mAP. For a local ONNX model, use `--model-path ./model.onnx`; pass `--framework onnx`
only when an explicit override is needed.

## Standard Multi-Model Runner

Use `benchmark_vision_models.py` for reproducible multi-model and core-mode sweeps. All models in
one invocation must use the same task and organized validation dataset. `--core-mode all` runs
`single`, `multi`, `global4`, and `global8` and writes one result row for each model/mode pair.

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models resnet18 resnet50 \
  --task image_classification \
  --core-mode all \
  --batch-size 8 \
  --data-path ~/.mblt_model_zoo/datasets/imagenet \
  --results-dir benchmark/vision/results/imagenet_core_modes
```

The runner writes the following reproducible artifacts:

- `results.json`: Schema-versioned machine-readable benchmark results.
- `results.csv`: Flat result rows for comparisons and spreadsheets.
- `results.md` and `summary.md`: Human-readable summary table and report.
- `accuracy.png`: Accuracy chart, unless `--no-plot` is provided.

Use `--collect-host-info` to include `mblt-tracker collect` output in the summary. A local
`--model-path`, `--mxq-path`, or `--onnx-path` applies to exactly one model target.

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

Use the standard runner. It records Top-1 as the primary score and Top-5 as the secondary metric in
the result artifacts.

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models resnet50 \
  --task image_classification \
  --mxq-path ./resnet50_IMAGENET1K_V1.mxq \
  --model-type IMAGENET1K_V1 \
  --core-mode global8 \
  --batch-size 8 \
  --data-path ~/.mblt_model_zoo/datasets/imagenet \
  --results-dir benchmark/vision/results/resnet50_imagenet
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

Use `object_detection`, `instance_segmentation`, or `pose_estimation` as appropriate for the selected model.

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models YOLOv5m \
  --task object_detection \
  --core-mode single \
  --batch-size 8 \
  --data-path ~/.mblt_model_zoo/datasets/coco \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --results-dir benchmark/vision/results/yolov5m_coco
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

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models yolo11n-face \
  --task face_detection \
  --batch-size 1 \
  --data-path ~/.mblt_model_zoo/datasets/widerface \
  --results-dir benchmark/vision/results/yolo11n_face_widerface
```

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

This keeps only the validation split and organizes the dataset into the following structure:

```text
~/.mblt_model_zoo/datasets/dotav1/
├── images/
│   ├── P0003.png
│   ├── P0006.png
│   └── ...
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

```bash
python benchmark/vision/benchmark_vision_models.py \
  --models YOLOv8s-obb \
  --task obb \
  --framework onnx \
  --batch-size 1 \
  --data-path ~/.mblt_model_zoo/datasets/dotav1 \
  --conf-thres 0.01 \
  --iou-thres 0.7 \
  --results-dir benchmark/vision/results/yolov8s_obb_dotav1
```

The benchmark records rotated `mAP test 50-95` as the primary score and `mAP test 50` as the
secondary metric; DOTA Task1 prediction files are stored under the corresponding `runs/`
directory.

## Organize NYU Depth Dataset

The NYU Depth organizer downloads the published archive by default and installs only its 654 validation image/depth
pairs under `images/` and `depth/` in the selected output directory. You can also give it a local ZIP file or
extracted dataset root; when `val` directories are present, training data is excluded.

```bash
python benchmark/vision/organize_nyu_depth.py \
  --output-dir ~/.mblt_model_zoo/datasets/nyu-depth
```

```text
~/.mblt_model_zoo/datasets/nyu-depth/
├── images/
│   ├── nyu_0000.jpg
│   └── ...
└── depth/
    ├── nyu_0000.npy
    └── ...
```

## Organize ADE20K Dataset

The ADE20K organizer downloads the official archive by default, then retains only the 2,000 validation image/mask
pairs. It installs them as flat `images/` and `annotations/` directories and preserves `objectInfo150.txt` and
`sceneCategories.txt` when supplied by the source.

```bash
python benchmark/vision/organize_ade20k.py \
  --output-dir ~/.mblt_model_zoo/datasets/ADEChallengeData2016
```

```text
~/.mblt_model_zoo/datasets/ADEChallengeData2016/
├── annotations/
│   ├── ADE_val_00000001.png
│   └── ...
├── images/
│   ├── ADE_val_00000001.jpg
│   └── ...
├── objectInfo150.txt
└── sceneCategories.txt
```

## Compare Vision Benchmark Results

You can compare multiple vision benchmark CSV files and generate model-wise charts:

```bash
python benchmark/vision/compare_benchmark_results.py \
  ./results/a4000 \
  ./results/a5000 \
  ./results/mla100
```

Output charts are saved under:

```text
benchmark/vision/results/charts/input1_input2_.../
```

Pass result directories or their `results.csv` files. Inputs must contain the same benchmark metric, and the
comparison chart includes only model/core-mode targets present in every source.
