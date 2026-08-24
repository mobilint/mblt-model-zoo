# Model Zoo Vision smoke tests

`tests/vision/` verifies the Model Zoo compatibility facade without duplicating
Vision implementation tests. The ordinary registry test is offline. The model
smoke test is skipped until explicitly enabled because it may download an MXQ
artifact and requires a configured NPU.

```bash
pytest -q tests/vision
pytest -q tests/vision --run-vision-smoke
```

Choose any supported model and board with pytest options:

```bash
pytest -q tests/vision --run-vision-smoke \
  --vision-model yolo11m-pose \
  --vision-target-device aries-rb \
  --vision-core-mode single
```

Use a local MXQ or ONNX file when needed:

```bash
pytest -q tests/vision --run-vision-smoke \
  --vision-model resnet50 \
  --vision-model-path /path/to/resnet50.onnx \
  --vision-framework onnx
```
