# Benchmark Scripts (Text Generation)

The `benchmark/` folder contains runnable scripts for text-generation benchmarking.

## Benchmark all available models

`benchmark/benchmark_text_generation_models.py` runs a prefill/decode sweep for every text-generation model returned by `mblt_model_zoo.hf_transformers.utils.list_models`, saves per-model JSON/PNG, and aggregates combined results.

```bash
python benchmark/benchmark_text_generation_models.py
```

Outputs (created under `benchmark/results/text_generation/`):
- `{model}.json` and `{model}.png` for each model
- `combined.png`, `combined.csv`, and `combined.md`

Common CLI options:
- `--device` (default: `cpu`)
- `--device-map`, `--dtype`, `--trust-remote-code/--no-trust-remote-code`
- `--revision` (e.g., `W8`)
- `--prefill-range` (e.g., `128:512:128`)
- `--decode-range` (e.g., `128:512:128`)
- `--fixed-decode` (default: `10`)
- `--fixed-prefill` (default: `128`)
- `--skip-existing` (skip models with existing outputs)

Example:

```bash
python benchmark/benchmark_text_generation_models.py \
  --revision W8 \
  --prefill-range 128:512:128 \
  --decode-range 128:512:128 \
  --skip-existing
```
