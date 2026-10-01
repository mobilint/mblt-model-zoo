# Benchmark Guide

Benchmark tooling now lives in the standalone packages that own each model family:

- Transformers throughput, latency, device-metric, ASR accuracy, and result-comparison tools are maintained by
  [transformers-mblt](https://github.com/mobilint/transformers-mblt/tree/main/benchmark/transformers). The
  `mblt-model-zoo tps` command runs the same `transformers-mblt tps` implementation and requires
  `pip install 'mblt-model-zoo[transformers]'`.
- Vision benchmarks, dataset organizers, and result comparison tools are maintained by
  [mblt-vision-python](https://github.com/mobilint/mblt-vision-python/tree/main/benchmark).

## Quick Vision CLI Validation

Use `mblt-model-zoo val` for a single-model, task-aware validation run. The command loads the
model, infers its task, selects the matching benchmark dataset, and reports the task metric. It
also prepares the default dataset layout automatically when needed.

```bash
mblt-model-zoo val --help
mblt-model-zoo val --model resnet50 --data-path ~/.mblt_model_zoo/datasets/imagenet
mblt-model-zoo val --model yolo11m --batch-size 8 --data-path ~/.mblt_model_zoo/datasets/coco
```

Use `--model-path` for a local MXQ or ONNX artifact, with `--framework` when the file extension
does not provide the desired framework explicitly:

```bash
mblt-model-zoo val \
  --model resnet50 \
  --model-path ./resnet50.mxq \
  --core-mode global8 \
  --data-path ~/.mblt_model_zoo/datasets/imagenet
```

For reproducible Vision benchmark or core-mode sweeps, use the
[standalone Vision benchmark runner](https://github.com/mobilint/mblt-vision-python/tree/main/benchmark),
which writes JSON, CSV, Markdown, and chart artifacts.

## Dataset and Result Handling

Benchmark datasets and model artifacts can be large. Keep downloaded datasets outside the
repository where possible, for example under `~/.mblt_model_zoo/datasets/`, and pass their paths
explicitly to organizer or benchmark commands. Do not commit downloaded datasets, model artifacts,
or generated benchmark results; `benchmark/**/results/` is ignored for this purpose.

For comparable results, record the model artifact or revision, runtime and hardware configuration,
batch size, and benchmark arguments alongside each run. Reuse the supplied organizers and
evaluators so dataset layouts and task metrics remain consistent with the published benchmarks.
