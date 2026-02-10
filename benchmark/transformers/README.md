# Benchmark Scripts (Text Generation)

The `benchmark/` folder contains runnable scripts for text-generation benchmarking.

## Benchmark all available models

`benchmark/transformers/benchmark_text_generation_models.py` runs a prefill/decode sweep for every text-generation model returned by `mblt_model_zoo.hf_transformers.utils.list_models`, saves per-model JSON/PNG, and aggregates combined results.

```bash
python benchmark/transformers/benchmark_text_generation_models.py
```

Outputs (created under `benchmark/transformers/results/text_generation/`):
- `{model}.json` and `{model}.png` for each model
- `combined.png`, `combined.csv`, and `combined.md`

Common CLI options:
- `--device` (default: `cpu`)
- `--device-map`, `--dtype`, `--trust-remote-code/--no-trust-remote-code`
- `--revision` (e.g., `W8`)
- `--all` (benchmark `W8` and `W4W8` branches only; skips main and adds `-W8`/`-W4V8` suffixes)
- `--prefill-range` (e.g., `128:512:128`)
- `--decode-range` (e.g., `128:512:128`)
- `--fixed-decode` (default: `10`)
- `--fixed-prefill` (default: `128`)
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
