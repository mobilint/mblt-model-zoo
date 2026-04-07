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

## Quick Start Guide

### Initializing a Vision Model

Vision models are loaded through `MBLT_Engine`. This is the same loading style used in
[`tests/vision`](tests/vision) and [`benchmark/vision`](benchmark/vision).

```python
from mblt_model_zoo.vision import MBLT_Engine

# Load a built-in model config.
# If mxq_path is empty, the MXQ file is downloaded from Hugging Face Hub and cached automatically.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    mxq_path="",
    core_mode="global8",
)

# Load a different recipe defined in the model YAML.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="IMAGENET1K_V1",
    mxq_path="",
    core_mode="global8",
)

# Load from a local MXQ file instead of downloading from the Hub.
model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    mxq_path="path/to/resnet50.mxq",
    core_mode="global8",
)
```

`MBLT_Engine` accepts these main arguments:

- `model_cls`: Model name or YAML config path.
- `model_type`: Variant key defined in the model YAML. `DEFAULT` resolves to the default entry in that file.
- `mxq_path`: Local MXQ path. Use `""` to download and cache the published MXQ automatically.
- `core_mode`: NPU execution mode. Supported values are `single`, `multi`, `global4`, and `global8`.

Model files are also available on our [HuggingFace Hub](https://huggingface.co/mobilint).

### Running Inference

You can pass an image path, PIL image, numpy array, or torch tensor to the preprocess pipeline.

```python
from mblt_model_zoo.vision import MBLT_Engine

model = MBLT_Engine(
    model_cls="resnet50",
    model_type="DEFAULT",
    mxq_path="",
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

For object detection models such as YOLO, pass task-specific thresholds to `postprocess()`:

```python
result = model.postprocess(output, conf_thres=0.5, iou_thres=0.5)
```

Available vision models are documented in [mblt_model_zoo/vision/README.md](mblt_model_zoo/vision/README.md).

## Model List

We provide models quantized with our advanced quantization techniques. You can check whether a model is available in our [configuration directory](mblt_model_zoo/vision/models/).

## Optional Extras

When working with tasks other than vision, extra dependencies may be required. Those options can be installed via `pip install mblt-model-zoo[NAME]` or `pip install -e .[NAME]`.

Currently, these optional functions are only available on environment equipped with Mobilint's [ARIES](https://www.mobilint.com/aries).

|Name|Use|Details|
|-------|------|------|
|transformers|For using HuggingFace transformers related models|[README.md](mblt_model_zoo/hf_transformers/README.md)|
|MeloTTS|For using MeloTTS models|[README.md](mblt_model_zoo/MeloTTS/README.md)|

For the `transformers` extra, the repository also includes:

- functional test instructions in [tests/transformers/TEST.md](tests/transformers/TEST.md)
- benchmark script usage in [benchmark/transformers/README.md](benchmark/transformers/README.md)

> Note: The `MeloTTS` extra includes `unidic`, which requires an additional dictionary download step. Python packaging (PEP 517/518) does not support running arbitrary post-install commands automatically, so run `mblt-unidic-download` (or `python -m unidic download`) after installing the extra when needed.

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
