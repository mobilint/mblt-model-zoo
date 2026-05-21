# Transformers Benchmark Usage Examples

The `benchmark/transformers/` directory and the `mblt-model-zoo tps` CLI provide speed measurement
tools for Hugging Face Transformers-based text-generation and image-text-to-text models. This guide
collects representative examples for quick single-model checks, model sweeps, result comparison, and
prefill chunk-size search workflows.

## Prerequisites

Install the Transformers integration and development tools:

```bash
pip install -e ".[transformers]" --group dev
```

Model loading may require Hugging Face Hub access, local `.mxq` files, the Mobilint NPU runtime, or
GPU drivers depending on the benchmark target. The validation commands in this document avoid model
inference and model downloads.

## Common Concepts

- `prefill`: Processes input tokens before generation. Common metrics are prefill TPS and TTFT.
- `decode`: Generates new tokens using the KV cache. Common metrics are decode TPS and decode duration.
- `core-mode`: Selects the NPU execution core configuration.
  - `single`: Uses `target_cores=["0:0"]`.
  - `global4`: Uses `target_clusters=[0]`.
  - `global8`: Uses `target_clusters=[0, 1]`.
  - `all`: Runs `single`, `global4`, and `global8` sequentially in benchmark scripts.
- Mobilint targets, including `mobilint/...`, `--mxq-path`, and `--mxq-dir`, default to
  `--device cpu` and `--device-backend npu` when those options are omitted. Explicit user values
  always take precedence.
- `device metrics`: Collects power, energy, utilization, and memory metrics.
  - `--device-backend npu`: Uses the Mobilint NPU tracker.
  - `--device-backend gpu`: Uses the GPU tracker.
  - `--device-backend auto`: Selects a tracker based on the model and device.
  - `--no-device-metrics`: Disables device metric collection.

## Quick CLI Usage

Use `mblt-model-zoo tps` when you want to quickly measure one model. The console entry point is
defined in `pyproject.toml` as `mblt-model-zoo = "mblt_model_zoo.cli:main"`.

### Show Help

```bash
mblt-model-zoo tps --help
mblt-model-zoo tps measure --help
mblt-model-zoo tps sweep --help
```

### Measure One Text-Generation Case

`measure` runs repeated measurements for one prefill/decode token configuration and prints summary
statistics.

```bash
mblt-model-zoo tps measure \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --prefill 512 \
  --decode 128 \
  --repeat 1 \
  --warmup 1 \
  --json benchmark/transformers/results/quick_measure.json
```

Representative output metrics:

- `prefill_tps`: Input-token processing speed.
- `decode_tps`: New-token generation speed.
- `ttft`: Time-to-first-token based on prefill latency.
- `decode_duration`: Decode phase duration.
- `avg_power`, `total_energy`, `avg_memory_used`: Device metrics when enabled.

### Sweep Prefill and Decode Cache Lengths

`sweep` measures several prefill lengths and decode cache lengths, then writes JSON, CSV, and PNG
outputs.

```bash
mblt-model-zoo tps sweep \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --prefill-range 128:512:128 \
  --cache-lengths 1024,2048,4096,8192 \
  --decode-window 128 \
  --repeat 1 \
  --warmup 1 \
  --plot benchmark/transformers/results/quick_sweep.png \
  --json benchmark/transformers/results/quick_sweep.json \
  --csv benchmark/transformers/results/quick_sweep.csv
```

Use `--no-plot` when you do not need a PNG plot.

```bash
mblt-model-zoo tps sweep \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --no-plot \
  --json benchmark/transformers/results/quick_sweep.json
```

### Measure with a Local `.mxq` File

Use `--mxq-path` to override the model's `.mxq` artifact during pipeline loading.

```bash
mblt-model-zoo tps measure \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --mxq-path ./local_mxq/Qwen2.5-1.5B-Instruct-W8.mxq \
  --device cpu \
  --core-mode global8 \
  --prefill 1024 \
  --decode 128 \
  --repeat 1
```

### Fix Prefill Chunk Size

Use `--prefill-chunk-size` to compare a fixed prefill chunk size across measurements.

```bash
mblt-model-zoo tps sweep \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --prefill-chunk-size 512 \
  --prefill-range 512:2048:512 \
  --cache-lengths 1024,2048,4096 \
  --decode-window 128 \
  --json benchmark/transformers/results/chunk512_sweep.json \
  --csv benchmark/transformers/results/chunk512_sweep.csv
```

### Measure an Original HF Model on GPU

For a non-Mobilint original Hugging Face model, use a CUDA device and the GPU tracker.

```bash
mblt-model-zoo tps measure \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --device-backend gpu \
  --device-gpu-id 0 \
  --prefill 512 \
  --decode 128 \
  --repeat 1 \
  --json benchmark/transformers/results/gpu_measure.json
```

### Sweep Image-Text-to-Text Models

`sweep --task image-text-to-text` uses synthetic image inputs and measures both the vision encode
stage and the LLM prefill/decode stage.

```bash
mblt-model-zoo tps sweep \
  --model mobilint/Qwen2-VL-2B-Instruct \
  --task image-text-to-text \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --llm-resolution 384 \
  --prefill-range 1024:4096:1024 \
  --cache-lengths 1024,2048,4096,8192 \
  --decode-window 128 \
  --prompt "Describe the image in one sentence." \
  --repeat 1 \
  --warmup 1 \
  --no-plot \
  --json benchmark/transformers/results/vlm_sweep.json \
  --csv benchmark/transformers/results/vlm_sweep.csv
```

VLM outputs include `vision_encode_ms`, `vision_fps`, `llm_prefill_tps`, `llm_decode_tps`, and
`llm_ttft_ms`.

## Benchmark Script Usage

The scripts under `benchmark/transformers/` are intended for multi-model runs, revision sweeps,
core-mode sweeps, result table generation, and chart generation.

### Benchmark Text-Generation Models

`benchmark_text_generation_models.py` runs a prefill sweep and a cache-length decode sweep for the
text-generation models returned by `mblt_model_zoo.hf_transformers.utils.list_models`.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --core-mode global8 \
  --prefill-range 128:512:128 \
  --cache-lengths 1024,2048,4096,8192 \
  --decode-window 128 \
  --warmup 1 \
  --skip-existing
```

`--model` accepts the Hugging Face repo id as-is, including `/` (for example,
`mobilint/Llama-3.2-1B-Instruct`). It benchmarks only that model when provided; when omitted, all
listed text-generation models are benchmarked.

Default output directory: `benchmark/transformers/results/text_generation/`.

- `{model}[-{revision}]-{core_mode}.json`: Per-model detailed benchmark payload.
- `{model}[-{revision}]-{core_mode}.png`: Per-model summary chart.
- `combined.csv`, `combined.md`: Combined model summary tables.
- `combined_device.csv`: Combined device metric summary.
- `prefill_tps.png`, `decode_tps.png`, `prefill_latency_ms.png`, `decode_duration_ms.png`: Core metric charts.
- `avg_power_w.png`, `total_energy_j.png`, `avg_utilization_pct.png`, `avg_memory_used_mb.png`: Device metric charts.

### Benchmark W8 and W4V8 Revisions

`--all` benchmarks only the `W8` and `W4V8` branches and skips the main branch.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --all \
  --core-mode global8 \
  --skip-existing
```

Use `--core-mode all` to compare all fixed core modes.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --all \
  --core-mode all \
  --skip-existing
```

This creates output files for each revision and core mode, for example:

```text
{model}-W8-single.json
{model}-W8-global4.json
{model}-W8-global8.json
{model}-W4V8-single.json
```

### Benchmark a Local `.mxq` Directory

Use `--mxq-dir` to benchmark only local `.mxq` files in a directory.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --mxq-dir ./local_mxq \
  --core-mode global8 \
  --prefill-range 512:2048:512 \
  --cache-lengths 1024,2048,4096,8192 \
  --decode-window 128 \
  --skip-existing
```

File names must follow the `<model_id>-<W8|W4V8>.mxq` pattern. A full repo id may encode `/` as
`__`.

```text
mobilint__Qwen2.5-1.5B-Instruct-W8.mxq
mobilint__Qwen2.5-1.5B-Instruct-W4V8.mxq
```

When `--mxq-dir` is set, `--original-models`, `--all`, and `--revision` are ignored. The revision is
read from the file name suffix.

### Benchmark Original HF Models for Comparison

Use `--original-models` to resolve listed Mobilint model ids to their parent/base model ids on the
Hugging Face Hub, then benchmark the unique parent ids. If `--device` is omitted, the script defaults
to `cuda:0` and `--device-backend auto`.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --original-models \
  --prefill-range 128:512:128 \
  --cache-lengths 1024,2048,4096 \
  --decode-window 128 \
  --skip-existing
```

### Benchmark Image-Text-to-Text Models

`benchmark_image_text_to_text_models.py` measures the vision stage and the LLM stage of
image-text-to-text models.

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --llm-resolution 384 \
  --llm-prefill-range 1024:4096:1024 \
  --llm-cache-lengths 1024,2048,4096,8192 \
  --llm-decode-window 128 \
  --repeat 1 \
  --warmup 1 \
  --skip-existing
```

Specify `--model` to benchmark a single model.

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py \
  --model mobilint/Qwen2-VL-2B-Instruct \
  --revision W8 \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --prefill-chunk-size 512 \
  --llm-prefill-range 1024:4096:1024 \
  --llm-cache-lengths 1024,2048,4096 \
  --skip-existing
```

Default output directory: `benchmark/transformers/results/image_text_to_text/`.

- `{model}[-{revision}]-{core_mode}.json`: Per-model full benchmark payload.
- `{model}[-{revision}]-{core_mode}.csv`: Per-run raw rows for `vision` and `llm`.
- `{model}[-{revision}]-{core_mode}.png`: Per-model summary chart.
- `combined.csv`, `combined.md`: Combined summary.
- `combined_llm.csv`, `combined_vision.csv`, `combined_device.csv`: Stage/device summaries.
- `llm_prefill_tps.png`, `llm_decode_tps.png`, `llm_ttft_ms.png`: LLM charts.
- `vision_encode_ms.png`, `vision_fps.png`: Vision charts.

### Rebuild Charts from Existing Results

Use `--rebuild-charts` to regenerate combined CSV, Markdown, and chart outputs from existing JSON
files without running benchmarks again.

```bash
python benchmark/transformers/benchmark_text_generation_models.py \
  --rebuild-charts
```

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py \
  --rebuild-charts
```

## Compare Result Folders

`plot_compare_benchmark_results.py` compares multiple benchmark result folders and generates
model-wise bar charts.

### Compare Text-Generation Results

```bash
python benchmark/transformers/plot_compare_benchmark_results.py \
  benchmark/transformers/results/MLA100/text_generation \
  benchmark/transformers/results/RTX3090/text_generation \
  --output-dir benchmark/transformers/results/charts/text_generation_compare \
  --task text-generation
```

### Compare VLM Results

```bash
python benchmark/transformers/plot_compare_benchmark_results.py \
  benchmark/transformers/results/MLA100/image_text_to_text \
  benchmark/transformers/results/RTX3090/image_text_to_text \
  --output-dir benchmark/transformers/results/charts/vlm_compare \
  --task image-text-to-text
```

If `--output-dir` is omitted, charts are saved under `benchmark/transformers/results/charts/` using
a directory name derived from the input folder names.

## Search Prefill Chunk Size

`search_prefill_chunk_size.py` searches candidate chunk sizes and selects the best value by prefill
TPS.

```bash
python benchmark/transformers/search_prefill_chunk_size.py \
  --mxq-dir ./local_mxq \
  --core-modes single,global4,global8 \
  --prefill-lengths 1024,2048 \
  --chunk-candidates 128,256,512,1024,2048 \
  --decode-length 16 \
  --time-guard-sec 300 \
  --repeat 1 \
  --warmup 1 \
  --skip-existing
```

If `--mxq-dir` is omitted, the script searches public `mobilint/` text-generation models for `W4V8`
and `W8` revisions. Default output directory: `benchmark/transformers/results/prefill_chunk_search/`.

- `records/*.json`: Detailed search records per model/core-mode.
- `all_measurements.csv`: All measured rows.
- `best_chunks.csv`: Best chunk per `(model, core_mode, prefill_length)`.
- `summary.json`: Run summary and skipped target information.
- `skipped_mxq_files.csv`: Skipped `.mxq` files and reasons.
- `failed_pairs.csv`: Failed measurement pairs.

Rebuild CSV and chart outputs from existing records without model loading:

```bash
python benchmark/transformers/search_prefill_chunk_size.py \
  --rebuild-charts
```

## Update Prefill Chunk-Size Configs

`update_prefill_chunk_size_configs.py` reads a CSV file and updates `npu_prefill_chunk_size` values in
Hugging Face `config.json` files. It runs in dry-run mode by default.

```bash
python benchmark/transformers/update_prefill_chunk_size_configs.py \
  --csv benchmark/transformers/prefill_chunk_size.csv
```

Limit the dry run to one model:

```bash
python benchmark/transformers/update_prefill_chunk_size_configs.py \
  --csv benchmark/transformers/prefill_chunk_size.csv \
  --model mobilint/Qwen2.5-1.5B-Instruct
```

Use `--apply` only when you intend to push config updates:

```bash
python benchmark/transformers/update_prefill_chunk_size_configs.py \
  --csv benchmark/transformers/prefill_chunk_size.csv \
  --apply
```

## Common Option Summary

### Pipeline and Model Loading

- `--model`: Model id or local path.
- `--tokenizer`: Tokenizer id or local path. Defaults to the model when omitted.
- `--revision`: Hugging Face Hub revision or branch, such as `W8` or `W4V8`.
- `--mxq-path`: Single local `.mxq` file override.
- `--mxq-dir`: Local `.mxq` directory for benchmark scripts.
- `--device`: Transformers pipeline device, such as `cpu` or `cuda:0`.
- `--device-map`: Transformers `device_map`, such as `auto`.
- `--dtype`: Data type, such as `auto`, `float16`, or `bfloat16`.
- `--trust-remote-code` / `--no-trust-remote-code`: Whether to trust HF remote code.

### Measurement Range

- `--prefill`, `--decode`: Single-case token counts for CLI `measure`.
- `--prefill-range`: Text-generation prefill sweep range in `start:end:step` format.
- `--cache-lengths`: Cache lengths for decode sweep.
- `--decode-window`: Decode token window measured at each cache length.
- `--image-resolutions`: Image resolutions for the VLM vision stage.
- `--prefill-range`, `--cache-lengths`, `--decode-window`: Text and VLM LLM-stage sweep ranges.
- `--repeat`: Number of measured repeats.
- `--warmup`: Number of warmup runs before measured runs.

### NPU/GPU Execution and Device Metrics

- `--core-mode`: One of `single`, `global4`, `global8`, or `all`. `all` is a benchmark-script sweep alias, not a model runtime core mode.
- `--target-cores`: Explicit target cores for the CLI, for example `"0:0;0:1;0:2;0:3"`.
- `--target-clusters`: Explicit target clusters for the CLI, for example `"0;1"`.
- `--device-metrics` / `--no-device-metrics`: Enable or disable device metric collection.
- `--device-backend`: One of `none`, `auto`, `gpu`, or `npu`.
- `--device-gpu-id`: GPU tracker target id, such as `0` or `0,1`.

### Result Management

- `--json`: JSON output path for CLI results.
- `--csv`: CSV output path for CLI sweep rows.
- `--plot`: PNG output path for CLI sweep plots.
- `--no-plot`: Disable CLI sweep plot output.
- `--results-dir`: Output directory for selected benchmark scripts.
- `--skip-existing`: Skip models that already have output files.
- `--rebuild-charts`: Rebuild CSV, Markdown, and chart outputs from existing JSON or record files.

## Safe Validation Commands

The following commands validate syntax and CLI option wiring without running model inference or
downloading models.

```bash
python -m py_compile \
  benchmark/common/chart_utils.py \
  benchmark/common/io_utils.py \
  benchmark/common/argparse_utils.py \
  benchmark/common/runtime_utils.py \
  benchmark/common/math_utils.py \
  benchmark/transformers/benchmark_text_generation_models.py \
  benchmark/transformers/benchmark_image_text_to_text_models.py \
  benchmark/transformers/search_prefill_chunk_size.py \
  benchmark/transformers/plot_compare_benchmark_results.py \
  benchmark/transformers/chart_utils.py
```

```bash
python benchmark/transformers/benchmark_text_generation_models.py --help
python benchmark/transformers/benchmark_image_text_to_text_models.py --help
python benchmark/transformers/search_prefill_chunk_size.py --help
python benchmark/transformers/plot_compare_benchmark_results.py --help
python benchmark/transformers/update_prefill_chunk_size_configs.py --help
mblt-model-zoo tps measure --help
mblt-model-zoo tps sweep --help
```
