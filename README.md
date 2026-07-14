# Mobilint Model Zoo

<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-model-zoo/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

**mblt-model-zoo** is a curated collection of AI models optimized by [Mobilint](https://www.mobilint.com/)’s Neural Processing Units (NPUs).

Designed to help developers accelerate deployment, Mobilint's Model Zoo offers access to public, pre-trained, and pre-quantized models for vision, language, and multimodal tasks. Along with performance results, we provide pre- and post-processing tools to help developers evaluate, fine-tune, and integrate the models with ease.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/mblt-model-zoo?logo=pypi&logoColor=white)](https://pypi.org/project/mblt-model-zoo/)
[![PyPI Downloads](https://static.pepy.tech/badge/mblt-model-zoo?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/mblt-model-zoo)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mblt-model-zoo?logo=python&logoColor=gold)](https://pypi.org/project/mblt-model-zoo/)

- Prepare environment equipped with Mobilint's NPU. In case you are not a Mobilint customer, please contact [us](mailto:tech-support@mobilint.com).
- Install **mblt-model-zoo** using pip:

```bash
pip install mblt-model-zoo
```

- If you want to install the latest version from the source, clone the repository and install it:

```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
pip install -e .
```

Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).

## Quick Start Guide

### Initializing a Vision Model

Vision models are loaded through `MBLT_Engine`. This is the same loading style used in
[`tests/vision`](tests/vision) and [`benchmark/vision`](benchmark/vision).

```python
from mblt_model_zoo.vision import MBLT_Engine

# Load a built-in model config.
# If model_path is empty, the MXQ file is downloaded from Hugging Face Hub and cached automatically.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    model_path="",
    core_mode="global8",
)

# Load a different recipe defined in the model YAML.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="IMAGENET1K_V1",
    model_path="",
    core_mode="global8",
)

# Load from a local MXQ file instead of downloading from the Hub.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    model_path="path/to/resnet50.mxq",
    core_mode="global8",
)

# Run a vision model with ONNX instead of MXQ.
# Local `.onnx` and `.mxq` paths auto-detect the framework from the suffix.
# If model_path is empty, the matching ONNX file is downloaded from the
# same Hugging Face repo and cached automatically.
model = MBLT_Engine(
    model_cls="alexnet",
    model_type="IMAGENET1K_V1",
    framework="onnx",
    model_path="",
)
```

`MBLT_Engine` accepts these main arguments:

- `model_cls`: Model name or YAML config path.
- `model_type`: Variant key defined in the model YAML. `DEFAULT` resolves to the default entry in that file.
- `framework`: Inference backend. When omitted, the engine infers `.mxq` and `.onnx` from `model_path` or `file_cfg.model_path` and otherwise falls back to `mxq`. `onnx` is supported for image classification, object detection, instance segmentation, and pose estimation.
- `model_path`: Local MXQ or ONNX path. Use `""` to download and cache the published artifact automatically for the selected framework.
- `mxq_path` and `onnx_path`: Backward-compatible explicit path aliases.
- `onnx_providers`: Optional ONNX Runtime provider order. By default, the engine prefers available GPU providers such as CUDA and falls back to CPU.
- `core_mode`: NPU execution mode. Supported values are `single`, `multi`, `global4`, and `global8`.

Model files are also available on our [HuggingFace Hub](https://huggingface.co/mobilint).

### Running Inference

You can pass an image path, PIL image, numpy array, or torch tensor to the preprocess pipeline.

```python
from mblt_model_zoo.vision import MBLT_Engine

model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    model_path="",
    core_mode="global8",
)

image_path = "path/to/image.jpg"

try:
    input_img = model.preprocess(image_path)
    output = model(input_img)
    result = model.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path="path/to/save/result.jpg",
        topk=5,
    )
finally:
    model.dispose()
```

For ONNX vision models, the preprocess and postprocess flow stays the same:

```python
from mblt_model_zoo.vision import MBLT_Engine

model = MBLT_Engine(
    model_cls="caformer_b36",
    framework="onnx",
)

try:
    input_img = model.preprocess("path/to/image.jpg")
    output = model(input_img)
    result = model.postprocess(output)
finally:
    model.dispose()
```

For object detection models such as YOLO, postprocessing thresholds are initialized from the model
YAML. You can either use those defaults directly:

```python
result = model.postprocess(output)
```

or override them once on the model before calling `postprocess()`:

```python
model.set_postprocess_thresholds(conf_thres=0.25)
result = model.postprocess(output)
```

Available vision models are documented in [mblt_model_zoo/vision/README.md](mblt_model_zoo/vision/README.md).

### Vision API Migration for 2.0.0

`mblt_model_zoo.vision` supports both the legacy top-level model imports and the task subpackage
imports. These imports are valid:

```python
from mblt_model_zoo.vision import ResNet50
from mblt_model_zoo.vision import YOLO11m
```

```python
from mblt_model_zoo.vision.image_classification import ResNet50
from mblt_model_zoo.vision.object_detection import YOLO11m
```

For new code, `MBLT_Engine` remains the preferred loading API:

```python
from mblt_model_zoo.vision import MBLT_Engine

model = MBLT_Engine(model_cls="resnet50", model_type="DEFAULT", model_path="", core_mode="global8")
```

The task subpackage imports remain available as compatibility wrappers around `MBLT_Engine`. You
can also inspect supported tasks and model names programmatically with `mblt_model_zoo.vision.list_tasks()`
and `mblt_model_zoo.vision.list_models()`.

For legacy class-style constructors, the old `product` argument is still accepted in `2.0.0` so
existing call sites do not fail immediately, but it is ignored by the YAML-backed model registry.
If you previously relied on `product` to choose a non-default artifact, migrate that selection to
explicit `model_cls`, `model_type`, or `model_path` values.

## Model List

We provide models quantized with our advanced quantization techniques. You can check whether a model is available in our [configuration directory](mblt_model_zoo/vision/models/).

## Optional Extras

When working with tasks other than vision, extra dependencies may be required. Those options can be installed via `pip install mblt-model-zoo[NAME]` or `pip install -e .[NAME]`.

Currently, these optional functions are only available on environment equipped with Mobilint's [ARIES](https://www.mobilint.com/aries).

|Name|Use|Details|
|-------|------|------|
|onnxruntime|For running vision models with `framework="onnx"` on CPU|Install with `pip install mblt-model-zoo[onnxruntime]`|
|onnxruntime-gpu|For running vision models with `framework="onnx"` with GPU-enabled ONNX Runtime|Install with `pip install mblt-model-zoo[onnxruntime-gpu]`|
|transformers|For using HuggingFace transformers related models|[README.md](mblt_model_zoo/hf_transformers/README.md)|
|MeloTTS|For using MeloTTS models|[README.md](mblt_model_zoo/MeloTTS/README.md)|
|qbcompiler|For generating mxq files with custom setting|[README.md](compile/README.md)|

For the `transformers` extra, the repository also includes:

- functional test instructions in [tests/transformers/TEST.md](tests/transformers/TEST.md)
- benchmark script usage in [benchmark/transformers/README.md](benchmark/transformers/README.md)

> Note: The `MeloTTS` extra includes `unidic`, which requires an additional dictionary download step. Python packaging (PEP 517/518) does not support running arbitrary post-install commands automatically, so run `mblt-unidic-download` (or `python -m unidic download`) after installing the extra when needed.

## Command Line Interface

Installing this package exposes the `mblt-model-zoo` console command:

```bash
mblt-model-zoo --help
```

The CLI provides Mobilint-specific helper commands and delegates selected upstream Hugging Face
Transformers commands to the installed `transformers` package.

### Vision Prediction And Validation

The vision CLI runs the same preprocess, NPU inference, postprocess, and plotting pipeline used by
the Python API. Use `predict` with a source image and a model name; the task is inferred from the
model configuration. `classify`, `detect`, `pose`, and `segment` are also accepted as aliases.

```bash
mblt-model-zoo predict --source ./cat.png --model resnet50
mblt-model-zoo predict --source ./street.jpg --model yolo11m --output ./result_detect.jpg
```

Vision commands accept a shared `--model-path` for local MXQ and local ONNX files. When
`--framework` is omitted, the CLI infers `.mxq` and `.onnx` suffixes and otherwise falls back to
MXQ. If the explicit framework conflicts with the local file suffix, the command fails with a
clear error. The compatibility flags `--mxq-path` and `--onnx-path` stay separate from
`--model-path`, so framework-specific resolution still works when both a local MXQ artifact and an
explicit ONNX runtime are involved.

```bash
mblt-model-zoo predict --source ./cat.png --model resnet50 --model-path ./resnet50.mxq
mblt-model-zoo predict --source ./cat.png --model resnet50 --model-path ./resnet50.onnx
mblt-model-zoo predict --source ./cat.png --model resnet50 --framework onnx
mblt-model-zoo predict --source ./cat.png --model resnet50 --framework onnx --mxq-path ./resnet50.mxq
mblt-model-zoo predict --source ./cat.png --model resnet50 --framework onnx --onnx-path ./resnet50.onnx
```

Prediction results are saved under `runs/vision/predict/` by default. Pass `--output` or
`--save-path` to choose a specific output file. Classification models accept `--topk`; object
detection, instance segmentation, and pose estimation models accept `--conf-thres` and
`--iou-thres`.

```bash
mblt-model-zoo predict --source ./cat.png --model resnet50 --topk 5
mblt-model-zoo predict --source ./street.jpg --model yolo11m --conf-thres 0.5 --iou-thres 0.5
```

Use `val` to validate a supported vision model on its benchmark dataset. Classification models use
ImageNet, while object detection, instance segmentation, and pose estimation models use COCO.
Validation also supports `--framework onnx`, the shared `--model-path` override, and the
framework-specific compatibility aliases.

```bash
mblt-model-zoo val --model resnet50
mblt-model-zoo val --model yolo11m --batch-size 8 --conf-thres 0.001 --iou-thres 0.7
mblt-model-zoo val --model resnet50 --model-path ./resnet50.mxq
mblt-model-zoo val --model resnet50 --model-path ./resnet50.onnx
mblt-model-zoo val --model resnet50 --framework onnx
mblt-model-zoo val --model resnet50 --framework onnx --mxq-path ./resnet50.mxq
mblt-model-zoo val --model resnet50 --framework onnx --onnx-path ./resnet50.onnx
```

Common NPU and artifact options are shared by the vision commands:

```bash
mblt-model-zoo predict \
  --source ./cat.png \
  --model resnet50 \
  --model-type DEFAULT \
  --model-path /path/to/model.mxq \
  --core-mode global8 \
  --dev-no 0
```

Use `--core-mode single`, `multi`, `global4`, or `global8` to select the NPU execution mode. For
manual placement, pass semicolon-separated values with `--target-cores`, such as `0:0;0:1`, or
`--target-clusters`, such as `0;1`. Full vision CLI details and supported model names are available
in [mblt_model_zoo/vision/README.md](mblt_model_zoo/vision/README.md).

### TPS Benchmark Helpers

The `tps` command measures token-per-second performance for Transformers-based text-generation and
image-text-to-text pipelines. It requires the `transformers` extra.

```bash
pip install "mblt-model-zoo[transformers]"
mblt-model-zoo tps measure --help
mblt-model-zoo tps sweep --help
```

Detailed TPS benchmark examples are available in
[benchmark/transformers/README.md](benchmark/transformers/README.md).

### MeloTTS Helpers

The `melo` command, also available as `melotts`, forwards arguments to the MeloTTS Click CLI. The
`melo-ui` command launches the MeloTTS Gradio WebUI. These commands require the `MeloTTS` extra.

```bash
pip install "mblt-model-zoo[MeloTTS]"
mblt-model-zoo melo --help
mblt-model-zoo melotts --help
mblt-model-zoo melo-ui --help
```

### Delegated Transformers Commands

When the first argument is one of `add-fast-image-processor`, `add-new-model-like`, `chat`,
`convert`, `download`, `env`, `run`, `serve`, or `version`, `mblt-model-zoo` delegates execution to
the installed Transformers CLI. For `chat` and `serve`, the CLI installs Mobilint model registration
hooks when the delegated Transformers backend loads models through the local serve command path.

## Verbose Option

By default, model initialization stays quiet. To print the model file size and MD5 hash whenever an MXQ model loads, set the environment variable `MBLT_MODEL_ZOO_VERBOSE` to a truthy value before running your script:

```bash
export MBLT_MODEL_ZOO_VERBOSE=true  # accepted values: true/1/yes/on (case-insensitive)
python your_script.py
```

### Example Verbose Output

```bash
Model Initialized
Model Size: 216.94 MB
Model Hash: 23c262c43b4c1c453dd0326e249480a0
Device Number: 0
Core Mode: single
Target Cores: [CoreId(cluster=Cluster.Cluster0, core=Core.Core0)]
Model Variant 0
        Input Shape: [(1, 200, 96), (1, 200, 96), (2, 200, 200)]
        Output Shape: [(1, 102400, 1)]
Model Variant 1
        Input Shape: [(1, 300, 96), (1, 300, 96), (2, 300, 300)]
        Output Shape: [(1, 153600, 1)]
Model Variant 2
        Input Shape: [(1, 400, 96), (1, 400, 96), (2, 400, 400)]
        Output Shape: [(1, 204800, 1)]
Model Variant 3
        Input Shape: [(1, 500, 96), (1, 500, 96), (2, 500, 500)]
        Output Shape: [(1, 256000, 1)]
Model Variant 4
        Input Shape: [(1, 600, 96), (1, 600, 96), (2, 600, 600)]
        Output Shape: [(1, 307200, 1)]
Model Variant 5
        Input Shape: [(1, 900, 96), (1, 900, 96), (2, 900, 900)]
        Output Shape: [(1, 460800, 1)]
```

Unset or set the variable to any other value to suppress these messages.

## License

The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

Additionally, the license for each model provided in this package follows the terms specified in the source link provided with it.

## Support & Issues

If you encounter any problems with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).
