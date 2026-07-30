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
- Use `obb` as the canonical vision task key for oriented bounding boxes while retaining
  `oriented_bounding_boxes` as a compatibility alias.
- Preserve anchorless decoded-output layout provenance through NMS. If a decoded tensor is
  ambiguous without provenance, prioritize raw channels-first normalization before candidates-first.
- Reuse `vision.utils.letterbox.LetterBoxGeometry` for forward resize/padding metadata and inverse
  dense-output crops. Keep image interpolation/padding and target interpolation/ignore padding
  task-specific.
- Require `pre_cfg.LetterBox` for YOLO detection postprocessors. Semantic validation loader
  callbacks must return `(image, metadata)` with `ratio_pad`; use the metadata-enabled preprocess path.
- Normalize dense MXQ outputs before inverse letterboxing: depth accepts single-image
  `[1, H/4, W/4]` maps and bilinearly upsamples them by four; Cityscapes semantic segmentation
  accepts `[H, W, 19]` or `[B, H, W, 19]` logits and converts them to NCHW before bilinear
  restoration and `argmax`. Preserve ONNX full-resolution depth, NCHW logits, and baked maps.
- Use an explicit default seed of `0` for new vision randomness.
- Keep qbcompiler optional and lazily imported; never add module-level qbcompiler imports or make
  it a base dependency.
- Keep `data_path`, `subset_path`, and `calib_data_path` mutually exclusive compilation entry
  levels; each skips earlier preparation stages.
- Resolve compilation datasets with both `post_cfg.task` and `post_cfg.dataset`. Dense calibration
  samples RGB files from NYU Depth, ADE20K, or Cityscapes `images/` directories.

## Datasets and Evaluation

- Preserve the NYU Depth organizer's 654 validation pairs as `images/` and `depth/` at the output
  root. For NYU Depth V2 evaluation, stretch inputs and targets to the configured model size,
  reject non-finite predictions at target-valid pixels before median alignment, pool those target-valid
  pixels, return `delta1` as the primary metric, and report `abs_rel` and `rmse` for diagnosis.
- Keep `eval_imagenet()` returning Top-1 as a float for compatibility, and use
  `eval_imagenet_metrics()` when structured Top-1 primary and Top-5 secondary metrics are needed.
  For DOTAv1, return rotated mAP50-95 as the primary metric and rotated mAP50 as the secondary metric.
- Keep structured evaluator results aligned on `primary_score` and `secondary_score`. Preserve
  numeric `eval_coco()` compatibility and use `eval_coco_metrics()` for mAP50-95 and mAP50.
- Use the standard benchmark runner for dense tasks: record `delta1`, `abs_rel`, and `rmse` for
  depth, and dispatch ADE20K or Cityscapes semantic evaluation from `post_cfg.dataset`.
- Preserve DOTAv1's 458 validation images directly under `images/`, with normalized and original
  label layouts. Keep loader compatibility with legacy `images/val` datasets.
- Preserve ADE20K's 2,000 validation image/mask pairs as flat `images/` and `annotations/` directories.
- Reuse ImageNet, COCO, DOTAv1, WiderFace, and dense validation or calibration roots only when
  taxonomy-specific layout, required metadata and targets, and the complete official split are valid.
- For ADE20K semantic validation, apply the same letterbox geometry to images and masks, pad masks with `255`, map
  source labels `1..150` to classes `0..149`, and report mIoU as primary with pixel accuracy secondary.
- For Cityscapes semantic validation, select exactly 500 paired validation images and `gtFine_labelIds` masks from the
  official ZIP packages, map canonical source IDs to classes `0..18`, pad ignored labels with `255`, and keep its
  pipeline independent from ADE20K. Reject other semantic validation taxonomies.
- Do not duplicate dataset URLs, paths, or long test commands owned by the registry or local guide.

## Validate Proportionately

- Install only the extra needed for the touched area.
- Run one targeted test file or `-k` selection first; append `-x` while fixing failures.
- Expect NPU tests and model-download tests to be environment-dependent. Report unavailable
  hardware, artifacts, or extras rather than broadening the test run.
- Run `pre-commit run --files <touched files>` for Python edits when hooks are available. For docs,
  run `git diff --check` and inspect rendered Markdown structure.
