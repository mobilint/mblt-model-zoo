# Vision Calibration Subsets

Use these scripts to create flat, deterministic image subsets for calibration. Each subset
directory contains selected image files only. Existing contents in `--output-dir` are removed
before the new selection is copied.

## Prepare the Source Dataset

The subset scripts do not download or extract datasets. Before creating a calibration subset, run
the matching organizer in [`benchmark/vision`](../../benchmark/vision/README.md). The organizers
download the default dataset archives when local paths are not supplied.

```bash
# ImageNet
python benchmark/vision/organize_imagenet.py \
  --output-dir ~/.mblt_model_zoo/datasets/imagenet

# COCO
python benchmark/vision/organize_coco.py \
  --output-dir ~/.mblt_model_zoo/datasets/coco

# DOTAv1
python benchmark/vision/organize_dotav1.py \
  --output-dir ~/.mblt_model_zoo/datasets/dotav1
```

To use archives or extracted datasets that you already downloaded, pass the local input paths to
the corresponding organizer as documented in the benchmark guide. Once organization finishes, use
the resulting dataset directory as `--data-dir` below.

## ImageNet

`make_imagenet_subset.py` selects the requested number of images from every category. Its default
of one image per category creates 1,000 images total.

```bash
python compile/vision/make_imagenet_subset.py \
  --data-dir ~/.mblt_model_zoo/datasets/imagenet \
  --output-dir /workspace/calib_imagenet \
  --subset-size 1 \
  --seed 0
```

## COCO

`make_coco_subset.py` selects validation images. The default subset size is 100.

```bash
python compile/vision/make_coco_subset.py \
  --data-dir ~/.mblt_model_zoo/datasets/coco \
  --output-dir /workspace/calib_coco \
  --subset-size 100 \
  --seed 0
```

## DOTAv1

`make_dotav1_subset.py` selects validation images. The default subset size is 100.

```bash
python compile/vision/make_dotav1_subset.py \
  --data-dir ~/.mblt_model_zoo/datasets/dotav1 \
  --output-dir /workspace/calib_dotav1 \
  --subset-size 100 \
  --seed 0
```

Using the same source dataset and seed produces the same selection.
