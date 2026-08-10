# Model Zoo Vision compatibility layer

Vision is maintained in
[mblt-vision-python](https://github.com/mobilint/mblt-vision-python). This Model
Zoo module preserves the historical `mblt_model_zoo.vision` import path for
existing applications.

For new code, install `mblt-vision-python` and import from `mblt_vision`:

```python
from mblt_vision import MBLT_Engine

model = MBLT_Engine(model_cls="resnet50")
```

The standalone package owns the model registry, model classes, preprocessing,
postprocessing, result types, model documentation, and Vision tests. See its
[Vision API guide](https://github.com/mobilint/mblt-vision-python/tree/main/mblt_vision)
for supported model families and runtime options.

Model Zoo continues to own its CLI, compilation, benchmark, and dataset-management
workflows.
