# Benchmark Scripts (Text Generation)

The `benchmark/` folder contains runnable scripts for text-generation benchmarking.

## Benchmark all available models

`benchmark/transformers/benchmark_text_generation_models.py` runs a prefill/decode sweep for every text-generation model returned by `mblt_model_zoo.hf_transformers.utils.list_models`, saves per-model JSON/PNG, and aggregates combined results.

```bash
python benchmark/transformers/benchmark_text_generation_models.py
```

Outputs (created under `benchmark/transformers/results/text_generation/`):

- `{model}[-{revision}]-{core_mode}.json` and `{model}[-{revision}]-{core_mode}.png` for each model
- `combined.csv` and `combined.md`
- `combined_device.csv` (when device metrics are available)
- metric-wise charts:
  - `prefill_tps.png`, `decode_tps.png`
  - `prefill_latency_ms.png`, `decode_duration_ms.png`
  - `avg_power_w.png`, `total_energy_j.png`
  - `avg_utilization_pct.png`, `p99_utilization_pct.png`
  - `avg_memory_used_mb.png`, `p99_memory_used_mb.png`
  - `avg_memory_used_pct.png`, `p99_memory_used_pct.png`
  - `prefill_tokens_per_j.png`, `decode_tokens_per_j.png`
  - `prefill_j_per_token.png`, `decode_j_per_token.png`

Common CLI options:

- `--device` (default: `None`)
- `--device-map`, `--dtype`, `--trust-remote-code/--no-trust-remote-code`
- `--revision` (e.g., `W8`)
- `--all` (benchmark `W8` and `W4V8` branches only; skips main and adds `-W8`/`-W4V8` suffixes)
- `--mxq-dir` (benchmark only local mxq files in a directory; filename pattern: `<model_id>-<W8|W4V8>.mxq`)
- `--prefill-range` (e.g., `128:512:128`)
- `--decode-range` (e.g., `128:512:128`)
- `--fixed-decode` (default: `10`)
- `--fixed-prefill` (default: `128`)
- `--prefill-chunk-size` (optional fixed prefill chunk size override)
- `--core-mode` (`single`, `global4`, `global8`, `all`; default: `global8`) for fixed-core benchmarking
- `--warmup` (default: `1`)
- `--original-models` (resolve listed Mobilint models to their parent/base model IDs on HF Hub, then benchmark unique parent IDs)
- `--device-metrics/--no-device-metrics` (default: `--device-metrics`)
- `--device-backend` (`none`, `auto`, `gpu`, `npu`; default: `none`)
- `--device-gpu-id` (e.g., `0` or `0,1`)
- `--cuda-precheck/--no-cuda-precheck` (best-effort VRAM pre-check before load; default: `--cuda-precheck`)
- `--cuda-precheck-margin` (default: `1.15`)
- `--skip-existing` (skip models with existing outputs)
- `--rebuild-charts` (skip benchmark run and rebuild `combined.csv`/`combined.md`/charts from existing JSON files)

Device auto-selection rule when `--device` is omitted:
- `--original-models` set: `--device` is auto-set to `cuda:0` and `--device-backend` to `auto`.
- `--original-models` not set: `--device` is auto-set to `cpu` and `--device-backend` to `npu`.

Example:

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --revision W8 \
  --core-mode global8 \
  --prefill-range 128:512:128 \
  --decode-range 128:512:128 \
  --skip-existing
```

Example (`--all`):

```bash
python benchmark/transformers/benchmark_text_generation_models.py --all --skip-existing
```

When `--all` or an explicit `--core-mode` is used, results are saved with suffixes in both the output files and table labels. When `--core-mode all` is used, each revision is benchmarked three times (`single`, `global4`, `global8`) against the same model or local `.mxq` file. For example:

- `{model}-W8-single.json`, `{model}-W8-single.png`
- `{model}-W8-global4.json`, `{model}-W8-global4.png`
- `{model}-W8-global8.json`, `{model}-W8-global8.png`
- `{model}-W4V8-single.json`, `{model}-W4V8-single.png`

Example (`--mxq-dir`):

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --mxq-dir ./local_mxq \
  --core-mode global8 \
  --skip-existing
```

Notes for `--mxq-dir`:
- Only files matching `<model_id>-<W8|W4V8>.mxq` are used.
- `<model_id>` can be full repo id (e.g. `mobilint/Qwen2.5-1.5B-Instruct`) or basename when uniquely resolvable.
- `--original-models`, `--all`, and `--revision` are ignored when `--mxq-dir` is set (revision is taken from filename suffix).
- A single local `.mxq` target is reused across every selected core mode. With `--core-mode all`, the script runs `single`, `global4`, and `global8` for each discovered `.mxq`.

## Benchmark image-text-to-text models

`benchmark/transformers/benchmark_image_text_to_text_models.py` benchmarks VLM models for:
- vision stage: encode latency / FPS across `--image-resolutions`
- llm stage: prefill/decode TPS at one reference resolution (`--llm-resolution`, default: first resolution)

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py
```

Common options:
- `--model` (benchmark a single model id)
- `--revision`, `--all` (W8/W4V8 sweep)
- `--mxq-dir` (benchmark only local mxq files in a directory; filename pattern: `<model_id>-<W8|W4V8>.mxq`)
- `--core-mode` (`single`, `multi`, `global4`, `global8`, `all`; default: `global8`)
- `--image-resolutions` (default: `224,384,512,768`)
- `--llm-resolution`, `--decode`, `--prompt`
- `--repeat`, `--warmup`
- `--original-models` (resolve listed Mobilint models to parent/base model IDs)
- `--device`, `--device-map`, `--dtype`, `--trust-remote-code`
- `--device-metrics/--no-device-metrics`, `--device-backend`, `--device-gpu-id`
- `--cuda-precheck/--no-cuda-precheck`, `--cuda-precheck-margin`
- `--skip-existing`, `--rebuild-charts`, `--results-dir`

Device auto-selection rule when `--device` is omitted:
- `--original-models` set: `--device` is auto-set to `cuda:0` and `--device-backend` to `auto`.
- `--original-models` not set: `--device` is auto-set to `cpu` and `--device-backend` to `npu`.

Outputs (default: `benchmark/transformers/results/image_text_to_text/`):
- `{model}[-{revision}]-{core_mode}.json`: per-model full benchmark payload
- `{model}[-{revision}]-{core_mode}.csv`: per-run raw rows (`vision`/`llm`)
- `{model}[-{revision}]-{core_mode}.png`: per-model summary chart
- `combined.csv`: model-wise integrated summary (LLM + vision)
- `combined.md`: markdown summary table
- `combined_device.csv`: model-wise device summary
- `combined_llm.csv`: model-wise LLM summary
- `combined_vision.csv`: per-model/resolution vision summary
- charts: `llm_prefill_tps.png`, `llm_decode_tps.png`, `llm_ttft_ms.png`, `vision_encode_ms.png`, `vision_fps.png`
- efficiency charts: `llm_prefill_tokens_per_j.png`, `llm_decode_tokens_per_j.png`, `llm_prefill_j_per_token.png`, `llm_decode_j_per_token.png`, `vision_img_per_j.png`, `vision_j_per_img.png`

Example:

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --decode 128 \
  --repeat 5 \
  --skip-existing
```

When `--core-mode all` is used, each target model or discovered local `.mxq` is benchmarked three times with `single`, `global4`, and `global8`, and the output filenames include the core-mode suffix.

## Compare result folders

You can compare multiple benchmark result folders and generate the same metric-wise chart set as benchmark output:

```bash
python benchmark/transformers/plot_compare_benchmark_results.py \
  <result_folder_1> <result_folder_2> [<result_folder_3> ...] \
  --output-dir benchmark/transformers/results/charts \
  --task text-generation
```

If `--output-dir` is omitted, charts are saved under:
`benchmark/transformers/results/charts/<sanitized_folder1_..._sanitized_folderN>/`

Expected per-model file format in each folder: `*.json` with top-level `"model"` and `"benchmark"` payload.
For backward compatibility, `group_id__model_id.json` is still parsed.
The script intersects model IDs across all folders, then saves:

- `prefill_tps.png`
- `decode_tps.png`
- `prefill_latency_ms.png`
- `decode_duration_ms.png`
- `avg_power_w.png`
- `total_energy_j.png`
- `prefill_tokens_per_j.png`
- `decode_tokens_per_j.png`
- `prefill_j_per_token.png`
- `decode_j_per_token.png`
- `avg_utilization_pct.png`
- `p99_utilization_pct.png`
- `avg_memory_used_mb.png`
- `p99_memory_used_mb.png`
- `avg_memory_used_pct.png`
- `p99_memory_used_pct.png`

For VLM (`image-text-to-text`) benchmark outputs, set `--task image-text-to-text`:

```bash
python benchmark/transformers/plot_compare_benchmark_results.py \
  <result_folder_1> <result_folder_2> [<result_folder_3> ...] \
  --task image-text-to-text
```

VLM compare mode saves:
- `llm_prefill_tps.png`
- `llm_decode_tps.png`
- `llm_ttft_ms.png`
- `llm_decode_duration_ms.png`
- `vision_encode_ms.png`
- `vision_fps.png`
- `vision_img_per_j.png`
- `vision_j_per_img.png`
- `llm_prefill_tokens_per_j.png`
- `llm_decode_tokens_per_j.png`
- `llm_prefill_j_per_token.png`
- `llm_decode_j_per_token.png`
- `avg_power_w.png`
- `total_energy_j.png`
- `avg_utilization_pct.png`
- `p99_utilization_pct.png`
- `avg_memory_used_mb.png`
- `p99_memory_used_mb.png`
- `avg_memory_used_pct.png`
- `p99_memory_used_pct.png`

## Search best prefill chunk size

`benchmark/transformers/search_prefill_chunk_size.py` searches the best `chunk_size` for **prefill TPS** while iterating valid `*.mxq` files in a directory and core mode (`single`, `global4`, `global8`).

Important behavior:
- Core mode is fixed at model creation time, so the script recreates the model instance for each core mode.
- Input targets come from `--mxq-dir` and filename pattern: `<model_id_without_group_id>-<W8|W4V8>.mxq`.
- If filename format is invalid, model base is not found in HF text-generation model list, or mapping is ambiguous, that mxq file is skipped with warning.
- The script sweeps fixed prefill lengths (`--prefill-lengths`, default: `1024,2048`).
- For each prefill length, it tests fixed chunk candidates (`--chunk-candidates`, default: `128,256,512,1024,2048`).
- Candidates where `chunk_size > prefill_length` are skipped automatically.
- Each `(prefill_length, chunk_size)` pair is measured with repeats and median prefill TPS.
- Runtime progress is shown with `tqdm` (overall pair progress, warmup, and per-pair search eval progress), plus search-stage logs.
- Time guard: chunks are tested in ascending order; if a chunk exceeds `--time-guard-sec` (default: `300s`), larger chunks are skipped for that prefill length.

Outputs (default: `benchmark/transformers/results/prefill_chunk_search/`):
- `records/*.json`: per model/core-mode detailed search history
- `all_measurements.csv`: all measured rows with `prefill_length`, `chunk_size`, TPS, wall time, and failure flags
- `best_chunks.csv`: best chunk per `(model, core_mode, prefill_length)`
- `summary.json`: run summary including skipped mxq files
- `skipped_mxq_files.csv`: skipped mxq file list and reasons
- `failed_pairs.csv`: pair-level runtime failures (if any)
- mode/prefill 2D charts:
  - `prefill_tps_single_prefill128_W8.png` (and `W4V8`, other prefill lengths)
  - `prefill_tps_global4_prefill128_W8.png` (and `W4V8`, other prefill lengths)
  - `prefill_tps_global8_prefill128_W8.png` (and `W4V8`, other prefill lengths)
  - each chart is revision-separated, and each line is `model_id@revision`
- best-chunk charts (by prefill length):
  - `best_chunk_single_W8.png` (and `W4V8`)
  - `best_chunk_global4_W8.png` (and `W4V8`)
  - `best_chunk_global8_W8.png` (and `W4V8`)

Example:

```bash
python benchmark/transformers/search_prefill_chunk_size.py \
  --mxq-dir . \
  --prefill-lengths 1024,2048 \
  --chunk-candidates 128,256,512,1024,2048 \
  --time-guard-sec 300 \
  --repeat 3 \
  --skip-existing
```
