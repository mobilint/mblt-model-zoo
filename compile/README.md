# Model Compilation

Vision compilation is maintained in `mblt-vision-python`. See the
[standalone Vision compilation guide](https://github.com/mobilint/mblt-vision-python/tree/main/compile/vision)
for dependency setup, automatic ONNX and dataset downloads, calibration defaults, and command helpers.

Registry-backed calibration supports ImageNet, COCO, WiderFace, DOTAv1, NYU Depth, ADE20K, and
Cityscapes. Dense models select NYU Depth, ADE20K, or Cityscapes from their `post_cfg.dataset`
metadata.

```python
from mblt_vision.compile import compile_vision_model

compile_vision_model("alexnet")
```

```bash
mblt-model-zoo compile --model-cls alexnet
```

By default, downloaded ONNX models and compiled MXQ outputs are stored under
`~/.mblt_model_zoo`, while registry-backed datasets are stored under
`~/.mblt_model_zoo/datasets`. Explicit model, data, calibration, and output paths still take
precedence.

## Optional Dependency Isolation

qbcompiler is not imported when the model-zoo package, vision APIs, compilation module, or main CLI
is imported. It is loaded only when `compile_vision_model()` or `mblt-model-zoo compile` starts a
compilation request. If qbcompiler is absent, that request exits with a concise installation error;
all non-compilation APIs and CLI commands remain available.

## Calibration Pipeline Entry Points

Compilation can begin from exactly one of three levels:

- `data_path`: Full organized image dataset; organize, sample, and preprocess it.
- `subset_path`: Already-sampled images; preprocess them without dataset preparation or sampling.
- `calib_data_path`: Ready preprocessed `.npy` tensors; validate and pass them directly to
  qbcompiler.

The CLI equivalents are `--data-path`, `--subset-path`, and `--calib-data-path`. See the vision
guide for tensor requirements and examples.
