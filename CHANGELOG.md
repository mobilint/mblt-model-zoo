# Changelog

## 2.3.0

### Changed

- The `mblt-model-zoo tps` CLI JSON schema is now driven by
  `mblt_model_zoo.cli.tps_table.TPS_TABLE_SPEC`.  The `runs`, `aggregate`,
  and `summary` blocks all use the same canonical key set as the printed
  CLI tables: bare unit-free names, optional `llm_` prefix for the LLM
  subtable inside VLM payloads, and a `_last` suffix in the `summary` block
  for sweep-only metrics.  Units now live in a top-level
  `llm_results.units` / `units` metadata block rather than being encoded
  into keys.  Old and new key mappings:

  | Old key | New canonical key |
  | --- | --- |
  | `avg_power_w` | `avg_power` |
  | `p99_power_w` | `p99_power` |
  | `avg_util_pct` | `avg_util` |
  | `p99_util_pct` | `p99_util` |
  | `avg_temp_c` | `avg_temp` |
  | `p99_temp_c` | `p99_temp` |
  | `avg_mem_used_mb` | `avg_mem_used` |
  | `p99_mem_used_mb` | `p99_mem_used` |
  | `avg_mem_used_pct` | `avg_mem_used_pct` (unchanged) |
  | `p99_mem_used_pct` | `p99_mem_used_pct` (unchanged) |
  | `total_memory_mb` / `total_mem_mb` | `total_mem` |
  | `prefill_energy_j` | `prefill_energy` |
  | `decode_energy_j` | `decode_energy` |
  | `vision_energy_j` | `vision_energy` |
  | `llm_total_energy_j` | `llm_total_energy` |
  | `total_energy_j` | `total_energy` |
  | `prefill_tps_last`, `decode_tps_last`, `ttft_ms_last`, `decode_duration_ms_last` | `prefill_tps_last`, `decode_tps_last`, `ttft_last`, `decode_duration_last` |
  | `prefill_npu_lat_pct_last`, `decode_npu_lat_pct_last`, `total_npu_lat_pct_last` | `prefill_npu_lat_last`, `decode_npu_lat_last`, `total_npu_lat_last` |
  | `prefill_tps_per_w_last`, `decode_tps_per_w_last`, `prefill_j_per_tok_last`, `decode_j_per_tok_last` | `prefill_tps_per_w`, `decode_tps_per_w`, `prefill_j_per_tok`, `decode_j_per_tok` (no `_last`; these are phase-wide aggregates, not sweep-last values) |
  | `vision_encode_ms` | `vision_encode` |
  | `accept_ratio_pct` | `accept_ratio` |

  VLM payloads gain the `llm_` prefix on all rows that have
  `llm_prefix=True` in `TPS_TABLE_SPEC` (e.g. `llm_avg_power`,
  `llm_prefill_energy`, `llm_prefill_tps`), keeping the printed CLI labels
  and JSON keys in lockstep.  The `_last` suffix in the summary block is
  reserved for rows with `sweep_suffix=True` (throughput / latency / NPU
  latency); phase-wide efficiency and device aggregates keep unsuffixed
  keys in every block.  A new `llm_results.units` block (and matching
  top-level `units` block for LLM measure / VLM measure) exposes the unit
  associated with each canonical key without encoding it into the key
  name.

  `llm_results.aggregate` and `llm_results.runs[i]` now expose every
  canonical row as a top-level key.  The nested `prefill_sweep` and
  `decode_sweep` blocks (raw x/tps/time/latency arrays) are still emitted
  verbatim; sweep-varying metrics like `llm_prefill_tps` now also appear
  as a top-level curve list in `aggregate` and `runs`.  `llm_total_energy`
  is sourced from `run.llm_total_energy_j` (LLM-only) and never falls back
  to `total_energy_j`, which now carries vision + LLM in VLM contexts.

  CLI printed tables also adopted the canonical short-form labels
  during this PR chain (introduced in `05174a7 Unify tps measure/sweep
  tables and auto-scale batch sweep lengths` and consolidated under
  `TPS_TABLE_SPEC` in `e60e702 Drive tps summary tables from a shared
  schema`).  Downstream tooling that parses printed CLI output should
  update to the new labels:

  | Old CLI label | New CLI label |
  | --- | --- |
  | `avg_utilization` / `avg_utilization_pct` (and `p99_`, `prefill_avg_`, `prefill_p99_`, `decode_avg_`, `decode_p99_`, `vision_avg_`, `vision_p99_` variants) | `avg_util` (and matching variants) |
  | `avg_temperature` / `avg_temperature_c` (and phase / vision variants) | `avg_temp` (and matching variants) |
  | `avg_memory_used` / `avg_memory_used_mb` (and phase / vision variants) | `avg_mem_used` (and matching variants) |
  | `total_memory` / `total_memory_mb` | `total_mem` |
  | `prefill_npu_latency` / `decode_npu_latency` / `total_npu_latency` (all with `_pct`) | `prefill_npu_lat` / `decode_npu_lat` / `total_npu_lat` |
  | `prefill_energy_j`, `decode_energy_j`, `vision_energy_j`, `total_energy_j`, `llm_total_energy_j` | `prefill_energy`, `decode_energy`, `vision_energy`, `total_energy`, `llm_total_energy` |
  | `avg_power_w` / `p99_power_w` (and phase / vision variants) | `avg_power` / `p99_power` (and matching variants) |
  | Sweep-last row suffix `(last_point)` | `(last)` |

  As with the JSON key changes, unit tokens (`_w`, `_c`, `_mb`, `_j`,
  and `_pct` where it merely restated the unit column) were stripped
  from CLI labels; units now live in the table's dedicated unit column
  and in the JSON `units` metadata block.  The `_pct` suffix is
  retained only where it disambiguates a percent-of metric from its
  absolute counterpart (`avg_mem_used` in MB vs. `avg_mem_used_pct` in
  %).  VLM tables prefix all shared LLM-phase rows with `llm_` (e.g.
  `llm_avg_util`, `llm_prefill_npu_lat`), matching the JSON key
  convention above.  Row labels for `ttft`, `decode_duration`,
  `prefill_tps`, `decode_tps`, `vision_encode`, `vision_fps`, and the
  EAGLE-3 acceptance rows are unchanged.

## 2.0.0

### Breaking Changes

- `mblt_model_zoo.vision` no longer re-exports legacy vision model classes at the package top level.
  Imports such as `from mblt_model_zoo.vision import ResNet50` and
  `from mblt_model_zoo.vision import YOLO11m` are no longer supported.
- Legacy `product` selection on compatibility model constructors is no longer functional in the
  YAML-backed vision registry. The argument is still accepted so older call sites do not fail at
  construction time, but it is ignored in `2.0.0`.
- The benchmark device-tracking integration now requires `mblt-tracker>=1.0.1`. The transformers
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
- Upgrade benchmark environments to `mblt-tracker>=1.0.1` before running transformers benchmark
  commands with device metrics enabled.
- Replace transformers compare-script invocations such as
  `python benchmark/transformers/plot_compare_benchmark_results.py ...` with
  `python benchmark/transformers/compare_benchmark_results.py ...` or
  `python -m benchmark.transformers.compare_benchmark_results ...`.
- Replace automatic speech recognition benchmark invocations that use `--model-id` or
  `--all-revisions` with `--model` and `--all`.
