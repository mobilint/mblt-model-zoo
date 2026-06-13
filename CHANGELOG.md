# Changelog

## 2.0.0

### Breaking Changes

- `mblt_model_zoo.vision` no longer re-exports legacy vision model classes at the package top level.
  Imports such as `from mblt_model_zoo.vision import ResNet50` and
  `from mblt_model_zoo.vision import YOLO11m` are no longer supported.
- Legacy `product` selection on compatibility model constructors is no longer functional in the
  YAML-backed vision registry. The argument is still accepted so older call sites do not fail at
  construction time, but it is ignored in `2.0.0`.
- The benchmark device-tracking integration now requires `mblt-tracker>=1.0.0`. The transformers
  benchmark tools use the tracker 1.x time-series APIs for power traces, NPU rail metrics, and
  trace-integrated energy values.
- The transformers benchmark comparison script was renamed from
  `benchmark/transformers/plot_compare_benchmark_results.py` to
  `benchmark/transformers/compare_benchmark_results.py`. The old transformers wrapper is no longer
  shipped.
- Transformers benchmark and CLI tokens-per-joule energy-efficiency fields were renamed to TPS/W.
  Result keys and plot filenames such as `prefill_tok_per_j`, `decode_tokens_per_j`, and
  `*_tokens_per_j.png` now use `prefill_tps_per_w`, `decode_tps_per_w`, and `*_tps_per_w.png`.
  Joules-per-token (`J/tok`) fields remain as a separate energy-efficiency metric.
- Automatic speech recognition benchmark flags were renamed: replace `--model-id` with `--model`
  and `--all-revisions` with `--all` in existing benchmark scripts.

### Changed

- Transformers benchmark energy and energy-efficiency metrics are now computed from mblt-tracker
  power traces with trapezoidal integration. At least two valid power samples are required, so very
  short runs can leave energy-derived fields empty.
- Transformers benchmark device tracking now supports NPU rail metric selection through
  `--device-npu-rail-metrics`, including `npu`, `ddr`, `pmic`, `goldfinger`, `all`, and
  comma-separated subsets.
- Transformers benchmark comparison output now supports text-generation, image-text-to-text, and
  automatic speech recognition result folders, including measure/sweep type detection, mixed-type
  rejection, source Host PC info summaries, and task-specific charts/tables.
- Transformers benchmark summaries and charts now label throughput-per-power efficiency as `TPS/W`
  while retaining `J/tok` as the inverse efficiency metric.
- Mobilint and non-Mobilint benchmark targets now resolve omitted runtime defaults per target, so a
  mixed target list can use NPU defaults for Mobilint targets and GPU defaults for Hugging Face
  targets while preserving explicit user-provided `--device` and `--device-backend` values.

### Migration Guide

- Prefer loading vision models through `mblt_model_zoo.vision.MBLT_Engine`.
- Legacy class-style imports remain available from task subpackages such as
  `mblt_model_zoo.vision.image_classification` and `mblt_model_zoo.vision.object_detection`.
- If older code used the legacy `product` argument to select non-default artifacts, migrate that
  selection to explicit `model_cls`, `model_type`, and `mxq_path` values.
- Use `mblt_model_zoo.vision.list_tasks()` and `mblt_model_zoo.vision.list_models()` to discover
  supported task and model names programmatically.
- Upgrade benchmark environments to `mblt-tracker>=1.0.0` before running transformers benchmark
  commands with device metrics enabled.
- Replace transformers compare-script invocations such as
  `python benchmark/transformers/plot_compare_benchmark_results.py ...` with
  `python benchmark/transformers/compare_benchmark_results.py ...` or
  `python -m benchmark.transformers.compare_benchmark_results ...`.
- Replace automatic speech recognition benchmark invocations that use `--model-id` or
  `--all-revisions` with `--model` and `--all`.
