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
- CLI defaults and benchmark-script defaults are aligned for shared TPS parameters.
  - `measure` defaults to `--prefill 128`, `--decode 32`, `--repeat 1`, and `--warmup 1`;
    VLM measure also defaults to `--image-resolution 224`.
  - `sweep` defaults to `--prefill-range 512:2048:512`,
    `--cache-lengths 128,512,1024,2048`, and `--decode-window 32`; VLM sweep also defaults
    to `--image-resolutions 224,384,512,768` and `--llm-resolution None`.
  - Benchmark scripts still support `--core-mode all` as a multi-run convenience; omitted
    `--core-mode` follows the CLI default of `None`.
- Benchmark scripts split Mobilint targets by config batch capability.
  - `--non-batch` is the default and benchmarks only targets whose config `max_batch_size` is `1`.
  - `--batch` benchmarks only targets whose config `max_batch_size` is greater than `1`.
  - Batch runs use the resolved `max_batch_size` exactly as the actual input batch size.
  - Batch text-generation sweep runs scale default `--prefill-range` and `--cache-lengths` down by
    `1/4`; explicit user-provided values are preserved. `--decode-window` is not scaled.
  - Batch TPS is total throughput across the batch: prefill tokens and decoded tokens are summed
    across all batch rows before dividing by elapsed time.
  - Models whose id contains `GGUF`, or whose local/Hub repository contains `.gguf` artifacts, are
    skipped by the Transformers benchmark scripts.
- `device metrics`: Collects power, energy, utilization, and memory metrics.
  - `--device-backend npu`: Uses the Mobilint NPU tracker.
  - `--device-backend gpu`: Uses the GPU tracker.
  - `--device-backend auto`: Selects a tracker based on the model and device.
  - `--device-npu-id 0,1`: Restricts NPU tracking to selected logical NPU card ids.
  - `--device-gpu-id 0,1`: Restricts GPU tracking to selected GPU ids.
  - `--no-device-metrics`: Disables device metric collection.
  - Console summaries show aggregate scalar metrics such as average and p99 power, utilization,
    temperature, and memory. JSON outputs also include device metric time-series under fields such
    as `device_time_series`, `device_time_series_runs`, or phase-specific `prefill`/`decode`
    entries. CSV outputs keep aggregate columns only.

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

When `--json` is set and device metrics are enabled, the JSON file includes per-run device
time-series for power, utilization, temperature, and memory. The time-series is not printed in the
summary table.

### Sweep Prefill and Decode Cache Lengths

`sweep` measures several prefill lengths and decode cache lengths, then writes JSON, CSV, and PNG
outputs.

```bash
mblt-model-zoo tps sweep \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --core-mode global8 \
  --prefill-range 512:2048:512 \
  --cache-lengths 128,512,1024,2048 \
  --decode-window 32 \
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

For Mobilint NPU runs, use `--device-npu-id` to restrict tracking to one or more logical NPU cards:

```bash
mblt-model-zoo tps measure \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --device cpu \
  --device-backend npu \
  --device-npu-id 0 \
  --prefill 512 \
  --decode 128 \
  --json benchmark/transformers/results/npu_measure.json
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

### Benchmark Automatic Speech Recognition Models

`benchmark_automatic_speech_recognition_models.py` evaluates Hugging Face Transformers
`automatic-speech-recognition` pipeline-compatible models on LibriSpeech and reports both accuracy
and speed metrics. Run the script twice with different `--num-beams` values to compare greedy
decoding and beam search.

The LibriSpeech loader uses streaming mode and consumes only the requested `--num-samples`, so the
benchmark does not eagerly download the entire evaluation split before running.

```bash
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py \
  --model-id mobilint/whisper-small \
  --revision W8 \
  --num-samples 5 \
  --num-beams 1 \
  --device cpu \
  --core-mode global8
```

```bash
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py \
  --model-id openai/whisper-small \
  --revision W8 \
  --num-samples 5 \
  --num-beams 5 \
  --device cpu \
  --core-mode global8
```

For Whisper-like models, `--language` and `--task` are passed as decoding hints. For other ASR
pipelines, the script automatically retries without those hints when they are unsupported.

Representative ASR metrics:

- `wer`, `cer`: Accuracy metrics computed from normalized transcripts.
- `mean_latency_s`, `p50_latency_s`, `p95_latency_s`: Per-sample generation latency.
- `throughput_samples_per_s`: Processed audio samples per second.
- `rtf`, `inverse_rtf`: Real-Time Factor and its inverse speed metric.
- `decode_tokens_per_s`: Decoder-side generated token throughput.
- Device metrics from `mblt-tracker` when enabled.

Original Hugging Face parent models can be benchmarked with `--original-models`:

```bash
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py \
  --original-models \
  --model-id mobilint/whisper-small mobilint/whisper-medium \
  --device cuda:0 \
  --dtype float16 \
  --device-backend gpu \
  --num-samples 5
```

You can also benchmark non-Whisper ASR models as long as they follow the Transformers ASR pipeline
contract:

```bash
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py \
  --model-id facebook/wav2vec2-base-960h \
  --num-samples 5 \
  --num-beams 1 \
  --device cuda:0 \
  --dtype float16 \
  --device-backend gpu
```

Local MXQ files can be discovered from a directory in the same style as the other benchmark
scripts:

```bash
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py \
  --mxq-dir ./local_mxq \
  --num-samples 5 \
  --num-beams 1
```

Outputs are written under `benchmark/transformers/results/automatic_speech_recognition/` by
default. Each run writes per-target JSON files with the beam count embedded in the filename, plus:

- `combined_beamsN.csv`
- `combined_beamsN.md`
- `summary_beamsN.md`
- optional charts such as `rtf_beamsN.png`, `wer_beamsN.png`, and `cer_beamsN.png`

Validation examples:

```bash
ruff check benchmark/transformers/benchmark_automatic_speech_recognition_models.py benchmark/transformers/asr_metrics.py tests/transformers/automatic_speech_recognition
pytest tests/transformers/automatic_speech_recognition/test_asr_metrics.py tests/transformers/automatic_speech_recognition/test_benchmark_asr_cli.py
python benchmark/transformers/benchmark_automatic_speech_recognition_models.py --help
```

### Benchmark Text-Generation Models

`benchmark_text_generation_models.py` requires a `measure` or `sweep` subcommand. `measure` runs a
fixed prefill/decode case across one or more models, while `sweep` runs a prefill sweep and a
cache-length decode sweep. Defaults match the TPS CLI.

```bash
python benchmark/transformers/benchmark_text_generation_models.py sweep \
  --non-batch \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --revision W8 \
  --core-mode global8 \
  --prefill-range 512:2048:512 \
  --cache-lengths 128,512,1024,2048 \
  --decode-window 32 \
  --warmup 1 \
  --skip-existing
```

To benchmark only batch-capable text-generation targets, pass `--batch`. The script uses each
target's config `max_batch_size` as the real input batch size and reports total token throughput.
For `sweep`, the default `--prefill-range` becomes `128:512:128` and default `--cache-lengths`
becomes `32,128,256,512`; `--decode-window` remains `32`. If you pass `--prefill-range` or
`--cache-lengths` explicitly, the script uses your values as-is.

```bash
python benchmark/transformers/benchmark_text_generation_models.py measure \
  --batch \
  --all \
  --prefill 512 \
  --decode 128 \
  --repeat 1 \
  --warmup 1
```

`--model` accepts the Hugging Face repo id as-is, including `/` (for example,
`mobilint/Llama-3.2-1B-Instruct`). It benchmarks only that model when provided; when omitted, all
listed text-generation models are benchmarked.

Default output directory: `benchmark/transformers/results/text_generation/`.

- `{model}[-{revision}]-{core_mode}.json`: Per-model detailed sweep payload.
- `{model}[-{revision}]-{core_mode}.png`: Per-model sweep summary chart.
- `{model}[-{revision}]-{core_mode}_measure.json`: Per-model measure payload.
- `combined_measure.csv`, `combined_measure.md`: Combined measure summary tables.
- `measure_prefill_tps.png`, `measure_decode_tps.png`: Measure charts.
- `combined.csv`, `combined.md`: Combined model summary tables.
- `combined_device.csv`: Combined device metric summary.
- `prefill_tps.png`, `decode_tps.png`, `prefill_latency_ms.png`, `decode_duration_ms.png`: Core metric charts.
- `avg_power_w.png`, `total_energy_j.png`, `avg_utilization_pct.png`, `avg_memory_used_mb.png`: Device metric charts.

### Benchmark W8 and W4V8 Revisions

`--all` benchmarks only the `W8` and `W4V8` branches and skips the main branch.

```bash
python benchmark/transformers/benchmark_text_generation_models.py sweep \
  --all \
  --core-mode global8 \
  --skip-existing
```

Use `--core-mode all` to compare all fixed core modes.

```bash
python benchmark/transformers/benchmark_text_generation_models.py sweep \
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
python benchmark/transformers/benchmark_text_generation_models.py sweep \
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
python benchmark/transformers/benchmark_text_generation_models.py sweep \
  --model mobilint/Qwen2.5-1.5B-Instruct \
  --original-models \
  --prefill-range 128:512:128 \
  --cache-lengths 1024,2048,4096 \
  --decode-window 128 \
  --skip-existing
```

### Benchmark Image-Text-to-Text Models

`benchmark_image_text_to_text_models.py` requires a `measure` or `sweep` subcommand. `measure` runs
one image resolution plus one LLM prefill/decode configuration; `sweep` measures the vision stage
over image resolutions and the LLM stage over prefill/cache ranges. Sweep option names match the TPS
CLI: `--prefill-range`, `--cache-lengths`, and `--decode-window`.

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py sweep \
  --non-batch \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --llm-resolution 384 \
  --prefill-range 1024:4096:1024 \
  --cache-lengths 1024,2048,4096,8192 \
  --decode-window 128 \
  --repeat 1 \
  --warmup 1 \
  --skip-existing
```

For batch-capable image-text-to-text targets, `--batch` uses config `max_batch_size` for the number
of synthetic images and text prompts. Vision FPS is reported as total images per second, and LLM
prefill/decode TPS is total token throughput across the batch.

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py measure \
  --batch \
  --model mobilint/Qwen2-VL-2B-Instruct \
  --revision W8 \
  --image-resolution 384 \
  --prefill 1024 \
  --decode 128 \
  --repeat 1 \
  --warmup 1
```

Specify `--model` to benchmark a single model.

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py sweep \
  --model mobilint/Qwen2-VL-2B-Instruct \
  --revision W8 \
  --core-mode global8 \
  --image-resolutions 224,384,512 \
  --prefill-chunk-size 512 \
  --prefill-range 1024:4096:1024 \
  --cache-lengths 1024,2048,4096 \
  --skip-existing
```

Default output directory: `benchmark/transformers/results/image_text_to_text/`.

- `{model}[-{revision}]-{core_mode}.json`: Per-model full sweep payload.
- `{model}[-{revision}]-{core_mode}.csv`: Per-run raw rows for `vision` and `llm`.
- `{model}[-{revision}]-{core_mode}.png`: Per-model sweep summary chart.
- `{model}[-{revision}]-{core_mode}_measure.json`: Per-model measure payload.
- `combined_measure.csv`, `combined_measure.md`: Combined measure summary tables.
- `measure_llm_prefill_tps.png`, `measure_llm_decode_tps.png`: Measure charts.
- `combined.csv`, `combined.md`: Combined summary.
- `combined_llm.csv`, `combined_vision.csv`, `combined_device.csv`: Stage/device summaries.
- `llm_prefill_tps.png`, `llm_decode_tps.png`, `llm_ttft_ms.png`: LLM charts.
- `vision_encode_ms.png`, `vision_fps.png`: Vision charts.

### Rebuild Charts from Existing Results

Use `--rebuild-charts` to regenerate combined CSV, Markdown, and chart outputs from existing JSON
files without running benchmarks again.

```bash
python benchmark/transformers/benchmark_text_generation_models.py sweep \
  --rebuild-charts
```

```bash
python benchmark/transformers/benchmark_image_text_to_text_models.py sweep \
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
- `--batch`, `--non-batch`: Benchmark-script target filters based on config `max_batch_size`.
  `--non-batch` is the default. `--batch` uses config `max_batch_size` exactly and reports total
  batch throughput.
- `--prefill-range`: Text-generation prefill sweep range in `start:end:step` format. This is also
  used by `mblt-model-zoo tps sweep --task image-text-to-text` for the VLM LLM-stage sweep.
- `--cache-lengths`: Cache lengths for decode sweep. This is also used by the TPS CLI VLM path.
- `--decode-window`: Decode token window measured at each cache length. This is also used by the TPS
  CLI VLM path.
- `--image-resolutions`: Image resolutions for the VLM vision stage.
- `--prefill-range`, `--cache-lengths`, `--decode-window`: LLM-stage sweep ranges for
  `benchmark_image_text_to_text_models.py`.
- `--repeat`: Number of measured repeats.
- `--warmup`: Number of warmup runs before measured runs.

### NPU/GPU Execution and Device Metrics

- `--core-mode`: One of `single`, `global4`, `global8`, or `all`. `all` is a benchmark-script sweep alias, not a model runtime core mode.
- `--target-cores`: Explicit target cores for the CLI, for example `"0:0;0:1;0:2;0:3"`.
- `--target-clusters`: Explicit target clusters for the CLI, for example `"0;1"`.
- `--device-metrics` / `--no-device-metrics`: Enable or disable device metric collection.
- `--device-backend`: One of `none`, `auto`, `gpu`, or `npu`.
- `--device-gpu-id`: GPU tracker target id, such as `0` or `0,1`.
- `--device-npu-id`: NPU tracker target logical card id, such as `0` or `0,1`.

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
downloading models. In this repository, use the project `uv` environment when available.

```bash
uv run python -m py_compile \
  benchmark/common/chart_utils.py \
  benchmark/common/io_utils.py \
  benchmark/common/argparse_utils.py \
  benchmark/common/runtime_utils.py \
  benchmark/common/math_utils.py \
  benchmark/transformers/benchmark_text_generation_models.py \
  benchmark/transformers/benchmark_image_text_to_text_models.py \
  benchmark/transformers/search_prefill_chunk_size.py \
  benchmark/transformers/plot_compare_benchmark_results.py \
  benchmark/transformers/update_prefill_chunk_size_configs.py \
  benchmark/transformers/chart_utils.py
```

```bash
uv run python benchmark/transformers/benchmark_text_generation_models.py --help
uv run python benchmark/transformers/benchmark_text_generation_models.py measure --help
uv run python benchmark/transformers/benchmark_text_generation_models.py sweep --help
uv run python benchmark/transformers/benchmark_image_text_to_text_models.py --help
uv run python benchmark/transformers/benchmark_image_text_to_text_models.py measure --help
uv run python benchmark/transformers/benchmark_image_text_to_text_models.py sweep --help
uv run python benchmark/transformers/search_prefill_chunk_size.py --help
uv run python benchmark/transformers/plot_compare_benchmark_results.py --help
uv run python benchmark/transformers/update_prefill_chunk_size_configs.py --help
uv run mblt-model-zoo tps measure --help
uv run mblt-model-zoo tps sweep --help
```
