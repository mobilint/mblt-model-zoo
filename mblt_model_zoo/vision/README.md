# Model Zoo Vision compatibility layer

Vision is maintained in
[mblt-vision-python](https://github.com/mobilint/mblt-vision-python). This Model
Zoo module preserves the historical `mblt_model_zoo.vision` import path for
existing applications. Its modules forward to `mblt_vision`; it contains no
independent model, processing, dataset, or evaluation implementation.

For new code, install `mblt-vision-python` and import from `mblt_vision`:

```python
from mblt_vision import MBLT_Engine

model = MBLT_Engine(model_cls="resnet50")
```

The standalone package owns the model registry, model classes, preprocessing,
postprocessing, result types, model documentation, and Vision tests. See its
[Vision API guide](https://github.com/mobilint/mblt-vision-python/tree/main/mblt_vision)
for supported model families and runtime options.

Model Zoo continues to host the `mblt-model-zoo predict`, `val`, and `compile`
commands for compatibility. Those commands load models, datasets, evaluators, and
compilation helpers from `mblt_vision` directly.

Legacy compatibility imports such as `mblt_model_zoo.vision.utils.Results` and
`mblt_model_zoo.vision.datasets.get_dataset_config` resolve to the standalone
implementation. New code should use their `mblt_vision` paths directly.
