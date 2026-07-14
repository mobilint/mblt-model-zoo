# Vision Model Compilation

The installable vision compiler uses each model's packaged YAML configuration and
`MBLT_Engine.preprocess` implementation to compile its ONNX artifact for ARIES.

## Installation

Install the package with the compilation extra:

```bash
pip install -e ".[qbcompiler]"
```

`qbcompiler` is currently distributed through Mobilint's external package channel and may not be
available from the public Python Package Index. Obtain access to the compiler package before using
this extra.

qbcompiler is loaded only when `compile_vision_model()` or `mblt-model-zoo compile` begins an
actual compilation request. Importing the base package, vision APIs, this compilation module, or
the main CLI does not import qbcompiler. If qbcompiler is unavailable, only the compilation request
fails with an installation message; non-compile APIs and CLI commands remain usable.

## Python API

```python
from mblt_model_zoo.compile.vision import compile_vision_model

output_path = compile_vision_model(
    "alexnet",
    data_path="~/.mblt_model_zoo/datasets/imagenet",
    save_path="./alexnet.mxq",
)
print(output_path)
```

## CLI

```bash
mblt-model-zoo compile \
  --model-cls alexnet \
  --data-path ~/.mblt_model_zoo/datasets/imagenet \
  --save-path ./alexnet.mxq
```

Start from an existing sampled image subset or a ready calibration tensor directory when those
stages have already been completed:

```bash
mblt-model-zoo compile --model-cls alexnet --subset-path ./sampled-images
mblt-model-zoo compile --model-cls alexnet --calib-data-path ./preprocessed-npy
```

Use `--model-type` for a non-default YAML variant and `--model-path` or `--onnx-path` to prefer a
local ONNX file. If that path is omitted or does not exist, the configured Hugging Face repository
supplies the ONNX artifact. Downloaded ONNX files and compiled outputs use
`~/.mblt_model_zoo`; without `--save-path`, the compiler writes
`~/.mblt_model_zoo/<onnx-stem>.mxq`.

## Calibration Data

The compilation data pipeline has three levels:

1. `data_path` / `--data-path`: Original organized dataset containing all images. The pipeline
   prepares the dataset when needed, samples a task-specific subset, and preprocesses it.
2. `subset_path` / `--subset-path`: Already-sampled images. The pipeline skips original dataset
   preparation and subset generation, then preprocesses every image in the supplied folder.
3. `calib_data_path` / `--calib-data-path`: Ready HWC, three-channel, contiguous `float32` `.npy`
   tensors. The directory is validated and passed directly to qbcompiler without image processing.

Supply at most one of these paths. When none is supplied, the task's registry-backed original
dataset path is used. `--calib-data-dir` remains an alias for `--calib-data-path` in the CLI and
standalone compatibility wrapper.

Compilation maps model tasks to the packaged dataset registry:

- Image classification uses ImageNet.
- Object detection, instance segmentation, and pose estimation use COCO.
- Face detection uses WiderFace.
- Oriented bounding boxes use DOTAv1.

`--data-path` is an organized dataset root. An existing ready layout is reused; otherwise the
registry-backed organizer downloads and prepares the dataset. When no path is supplied, the
registry default under `~/.mblt_model_zoo/datasets` is used.

Selection is deterministic with seed `0`. ImageNet and WiderFace select one image from every
category subfolder by default; COCO and DOTAv1 select 100 images total. Override these values with
`--subset-size` and `--seed`. Selected images and preprocessed NumPy arrays live only in temporary
directories and are removed after compilation, including when compilation fails.

The compiler uses explicit `--percentile` and `--topk-ratio` values first. Missing values are read
independently from `aries/best_result.json`; if optional hosted values are unavailable, defaults of
`0.9999` and `0.01` are used with a warning.

## Compatibility Scripts

The standalone scripts remain available:

```bash
python compile/vision/vision_model_compile.py --model-cls alexnet
python compile/vision/make_imagenet_subset.py --output-dir ./imagenet-calibration
python compile/vision/make_coco_subset.py --output-dir ./coco-calibration
python compile/vision/make_dotav1_subset.py --output-dir ./dotav1-calibration
python compile/vision/make_widerface_subset.py --output-dir ./widerface-calibration
```
