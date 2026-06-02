# Changelog

## 2.0.0

### Breaking Changes

- `mblt_model_zoo.vision` no longer re-exports legacy vision model classes at the package top level.
  Imports such as `from mblt_model_zoo.vision import ResNet50` and
  `from mblt_model_zoo.vision import YOLO11m` are no longer supported.
- Legacy `product` selection on compatibility model constructors is no longer functional in the
  YAML-backed vision registry. The argument is still accepted so older call sites do not fail at
  construction time, but it is ignored in `2.0.0`.

### Migration Guide

- Prefer loading vision models through `mblt_model_zoo.vision.MBLT_Engine`.
- Legacy class-style imports remain available from task subpackages such as
  `mblt_model_zoo.vision.image_classification` and `mblt_model_zoo.vision.object_detection`.
- If older code used the legacy `product` argument to select non-default artifacts, migrate that
  selection to explicit `model_cls`, `model_type`, and `mxq_path` values.
- Use `mblt_model_zoo.vision.list_tasks()` and `mblt_model_zoo.vision.list_models()` to discover
  supported task and model names programmatically.
