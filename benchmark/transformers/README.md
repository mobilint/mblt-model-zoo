# Benchmark Scripts (Text Generation)

The `benchmark/` folder contains runnable scripts for text-generation benchmarking.

## Benchmark all available models

`benchmark/transformers/benchmark_text_generation_models.py` runs a prefill/decode sweep for every text-generation model returned by `mblt_model_zoo.hf_transformers.utils.list_models`, saves per-model JSON/PNG, and aggregates combined results.

```bash
python benchmark/transformers/benchmark_text_generation_models.py
```

Outputs (created under `benchmark/transformers/results/text_generation/`):

- `{model}.json` and `{model}.png` for each model
- `combined.csv` and `combined.md`
- `combined_device.csv` (when device metrics are available)
- metric-wise charts:
  - `prefill_tps.png`, `decode_tps.png`
  - `prefill_latency_ms.png`, `decode_duration_ms.png`
  - `avg_power_w.png`, `total_energy_j.png`
  - `prefill_tok_per_j_last.png`, `decode_tok_per_j_last.png`
  - `prefill_j_per_tok_last.png`, `decode_j_per_tok_last.png`

Common CLI options:

- `--device` (default: `None`)
- `--device-map`, `--dtype`, `--trust-remote-code/--no-trust-remote-code`
- `--revision` (e.g., `W8`)
- `--all` (benchmark `W8` and `W4W8` branches only; skips main and adds `-W8`/`-W4V8` suffixes)
- `--prefill-range` (e.g., `128:512:128`)
- `--decode-range` (e.g., `128:512:128`)
- `--fixed-decode` (default: `10`)
- `--fixed-prefill` (default: `128`)
- `--warmup` (default: `1`)
- `--original-models` (resolve listed Mobilint models to their parent/base model IDs on HF Hub, then benchmark unique parent IDs)
- `--device-metrics/--no-device-metrics` (default: `--device-metrics`)
- `--device-backend` (`auto`, `gpu`, `npu`)
- `--device-interval` (default: `0.2`)
- `--device-gpu-id` (e.g., `0` or `0,1`)
- `--cuda-precheck/--no-cuda-precheck` (best-effort VRAM pre-check before load; default: `--cuda-precheck`)
- `--cuda-precheck-margin` (default: `1.15`)
- `--skip-existing` (skip models with existing outputs)

Example:

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --revision W8 \
  --prefill-range 128:512:128 \
  --decode-range 128:512:128 \
  --skip-existing
```

Example (`--all`):

```bash
python benchmark/transformers/benchmark_text_generation_models.py --all --skip-existing
```

When `--all` is used, results are saved with suffixes in both the output files and table labels, for example:

- `{model}-W8.json`, `{model}-W8.png`
- `{model}-W4V8.json`, `{model}-W4V8.png`

## Compare result folders

You can compare multiple benchmark result folders and generate 6 charts:

```bash
python benchmark/transformers/plot_compare_benchmark_results.py \
  <result_folder_1> <result_folder_2> [<result_folder_3> ...] \
  --output-dir benchmark/transformers/results/charts
```

If `--output-dir` is omitted, charts are saved under:
`benchmark/transformers/results/compare/<dirname1_dirname2_...>/`

Expected per-model file format in each folder: `group_id__model_id.json`.
The script ignores `group_id`, intersects model IDs across all folders, then saves:

- `prefill_tps.png`
- `decode_tps.png`
- `prefill_tokens_per_j.png`
- `decode_tokens_per_j.png`
- `prefill_j_per_token.png`
- `decode_j_per_token.png`

