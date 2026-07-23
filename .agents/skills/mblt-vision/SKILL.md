---
name: mblt-vision
description: >-
  Work effectively on Mobilint Model Zoo vision models, datasets, evaluation, prediction, validation,
  benchmarks, and optional qbcompiler compilation while preserving public compatibility contracts.
---

# Mobilint Model Zoo Vision

## Start Here

1. Read `AGENTS.md` and the shared `mblt-model-zoo` skill.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml`, `tests/vision/TEST.md`, and `benchmark/vision/README.md`. For compilation,
   also read `compile/vision/README.md`.

## Use the Correct APIs

- Use `mblt_model_zoo.vision.MBLT_Engine` for new vision loading code. Prefer `model_path`; treat
  `mxq_path` and `onnx_path` as compatibility aliases.
- Update task-package exports and lazy top-level vision exports together when adding or renaming a
  vision model. Confirm `list_models()` discovery.
- Use the YAML dataset registry and `get_dataset_config_for_task()` for dataset defaults.
- Use `compile_vision_model()` or the `compile` CLI only after the `qbcompiler` extra is installed.

## Preserve Contracts

- Keep legacy vision constructor arguments and imports working unless the task explicitly changes
  compatibility.
- Preserve automatic `.mxq`/`.onnx` framework detection and errors for conflicting explicit
  framework selections.
- Treat `file_cfg.filename` as the canonical MXQ Hub artifact and derive its same-stem ONNX
  artifact. Use `onnx_filename` only when the published ONNX artifact has a different filename.
- Require `post_cfg.dataset` in every model YAML and resolve class counts from the dataset-task
  pair. Do not assume all models for one task share a taxonomy.
- Use an explicit default seed of `0` for new vision randomness.
- Keep qbcompiler optional and lazily imported; never add module-level qbcompiler imports or make
  it a base dependency.
- Keep `data_path`, `subset_path`, and `calib_data_path` mutually exclusive compilation entry
  levels; each skips earlier preparation stages.

## Datasets and Evaluation

- Preserve the NYU Depth organizer's 654 validation pairs as `images/` and `depth/` at the output
  root. For NYU Depth V2 evaluation, stretch inputs and targets to the configured model size,
  median-align each prediction, pool valid pixels, return `delta1` as the primary metric, and
  report `abs_rel` and `rmse` for diagnosis.
- For ImageNet, return Top-1 accuracy as the primary metric and Top-5 accuracy as the secondary
  metric. For DOTAv1, return rotated mAP50-95 as the primary metric and rotated mAP50 as the
  secondary metric.
- Preserve DOTAv1's 458 validation images directly under `images/`, with normalized and original
  label layouts.
- Preserve ADE20K's 2,000 validation image/mask pairs as flat `images/` and `annotations/` directories.
- Do not duplicate dataset URLs, paths, or long test commands owned by the registry or local guide.

## Validate Proportionately

- Install only the extra needed for the touched area.
- Run one targeted test file or `-k` selection first; append `-x` while fixing failures.
- Expect NPU tests and model-download tests to be environment-dependent. Report unavailable
  hardware, artifacts, or extras rather than broadening the test run.
- Run `pre-commit run --files <touched files>` for Python edits when hooks are available. For docs,
  run `git diff --check` and inspect rendered Markdown structure.
