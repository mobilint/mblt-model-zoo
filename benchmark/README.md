# Benchmark Guide

The `benchmark/` directory contains reproducible performance and accuracy evaluation tools for
models in this repository. Each benchmark family owns its dataset preparation, command-line
options, result formats, and detailed usage instructions.

## Directory Layout

- [`vision/`](vision/README.md): Dataset organizers and accuracy benchmarks for vision models.
- [`transformers/`](transformers/README.md): Throughput, latency, device-metric, and result
  comparison tools for Transformers-based models.
- `common/`: Shared utilities for argument parsing, dataset handling, runtime setup, summaries,
  charts, and file I/O.

## Running Benchmarks

Run benchmark commands from the repository root so local imports and the documented relative paths
resolve consistently. Install the project and the extras required by the benchmark family before
running a command:

```bash
pip install -e . --group dev
```

The target model, dataset, and hardware runtime determine any additional requirements. Use the
family-specific guide for setup, supported commands, and examples:

- [Vision benchmarks](vision/README.md)
- [Transformers benchmarks](transformers/README.md)

## Dataset and Result Handling

Benchmark datasets and model artifacts can be large. Keep downloaded datasets outside the
repository where possible, for example under `~/.mblt_model_zoo/datasets/`, and pass their paths
explicitly to organizer or benchmark commands. Do not commit downloaded datasets, model artifacts,
or generated benchmark results; `benchmark/**/results/` is ignored for this purpose.

For comparable results, record the model artifact or revision, runtime and hardware configuration,
batch size, and benchmark arguments alongside each run. Reuse the supplied organizers and
evaluators so dataset layouts and task metrics remain consistent with the published benchmarks.
