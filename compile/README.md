# Model Compilation

Vision models can be compiled through the installed Python API or CLI. See the
[vision compilation guide](vision/README.md) for dependency setup, automatic ONNX and dataset
downloads, calibration defaults, and compatibility scripts.

```python
from mblt_model_zoo.compile.vision import compile_vision_model

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
