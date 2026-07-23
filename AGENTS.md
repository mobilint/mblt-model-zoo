---
description: Shared repository guidance for Codex, Claude Code, and other coding agents working on Mobilint Model Zoo.
paths:
  - "**"
---

# Mobilint Model Zoo Agent Guide

## Purpose and Precedence

This is the canonical repository guide. `CLAUDE.md` imports it so Codex and Claude Code use the
same rules. Shared workflow lives in `.agents/skills/mblt-model-zoo`; focused Vision and
Transformers skills live alongside it. Claude Code loads matching small entry points from
`.claude/skills/`. Follow more-specific `AGENTS.md` files in a subdirectory when present. User and
system instructions take precedence.

Keep this guide and `CLAUDE.md` synchronized for shared repository guidance so Codex and Claude
Code receive the same workflow requirements. The focused reusable skills are
`.agents/skills/mblt-vision` and `.agents/skills/mblt-transformers`; maintain matching Claude
entry points in `.claude/skills/`.

Before editing, run `git status --short`; preserve unrelated changes in a dirty worktree.

## Current Package Snapshot

- Package: `mblt-model-zoo` 2.2.1; Python `>=3.10,<3.13`.
- Runtime dependencies are declared in `pyproject.toml`; install development tools with
  `pip install -e . --group dev`.
- Optional extras: `transformers`, `MeloTTS`, `onnxruntime`, `onnxruntime-gpu`, `qwen-asr`, and
  `qbcompiler`.
- Console scripts: `mblt-model-zoo` and `mblt-melotts-download`.
- The main CLI provides `predict` (aliases: `classify`, `detect`, `pose`, `segment`), `val`,
  `compile`, `tps`, `melo` (alias: `melotts`), and `melo-ui`. It delegates supported upstream
  Transformers commands when that extra is installed.

Treat `pyproject.toml`, CLI parsers, public package exports, and area README files as the source of
truth when this snapshot becomes stale.

## Repository Map

- `mblt_model_zoo/vision`: public vision API, model wrappers, dataset registry, evaluation, and
  task packages.
- `mblt_model_zoo/compile`: installable compilation API; vision compilation is optional.
- `mblt_model_zoo/hf_transformers`: Hugging Face integrations and benchmark utilities.
- `mblt_model_zoo/MeloTTS`: MeloTTS integration and text normalization.
- `mblt_model_zoo/cli`: installed CLI implementation.
- `tests`: pytest suites and shared NPU option helpers.
- `benchmark`: vision and Transformers benchmark scripts; `compile/` holds compatibility scripts
  and compilation documentation.

## General Engineering Rules

- Use four-space indentation, PEP 484 annotations, and Google-style docstrings for new or modified
  Python modules, classes, functions, and methods.
- Keep lines at 120 characters or fewer. Let Ruff manage imports and formatting.
- Group imports as standard library, third-party, then local.
- Catch specific exceptions and provide recovery-oriented error messages. Do not catch `Exception`
  unless immediately re-raising or deliberately adding context.
- Preserve local style in `mblt_model_zoo/hf_transformers` and `mblt_model_zoo/MeloTTS`: both are
  excluded from repository-wide Ruff checks.
- Write comments for non-obvious rationale, not mechanics. Format temporary notes as
  `TODO(username): description`.

## Area-Specific Contracts

### Vision

- Prefer `MBLT_Engine` and task-subpackage imports in new code. Legacy top-level model imports
  remain supported and must stay synchronized with task-package `__all__` exports so
  `vision.list_models()` continues to work.
- Preserve public compatibility arguments such as `local_path`, `model_type`, `infer_mode`, and
  `product`. The YAML registry ignores `product`; select a non-default artifact with `model_cls`,
  `model_type`, or `model_path` instead.
- Prefer `model_path` in new APIs, tests, and docs. `mxq_path` and `onnx_path` are compatibility
  aliases. Framework inference recognizes local `.mxq` and `.onnx` suffixes; retain the fail-fast
  error for an explicit framework that conflicts with a local suffix.
- In model YAML `file_cfg`, use `filename` as the canonical MXQ artifact and let the loader derive
  the same-stem ONNX filename. Set `onnx_filename` only for a Hub artifact that has a different name.
- Every model YAML `post_cfg` must declare `dataset` as the output taxonomy and validation-dataset
  identifier. Dataset-aware postprocessing resolves class counts from the `(dataset, task)` pair.
- The supported discovery tasks are `image_classification`, `depth_estimation`, `object_detection`,
  `instance_segmentation`, `semantic_segmentation`, `oriented_bounding_boxes`, `obb`, `pose_estimation`, and
  `face_detection`. `obb` is an alias for `oriented_bounding_boxes`.
- Keep model configuration shape (`model_cfg`, `pre_cfg`, and `post_cfg`) stable unless changing
  the public contract deliberately.

### Vision Datasets and Compilation

- Keep validation datasets in `mblt_model_zoo/vision/datasets/*.yaml`. Use `path`, `val`, optional
  `names`, and the repository `tasks` and `download` metadata; resolve defaults with
  `get_dataset_config_for_task()` rather than duplicating URLs or paths.
- Keep the NYU Depth organizer validation-only: install its 654 paired samples as `images/` and `depth/` directly
  under its output root.
- Keep the ADE20K organizer validation-only: install its 2,000 paired samples as flat `images/` and `annotations/`
  directories, along with its source metadata files.
- Depth-estimation validation stretches RGB and depth targets to the configured input size, median-aligns each
  prediction, pools statistics over valid NYU Depth V2 pixels, and reports `delta1` as the primary score with
  `abs_rel` and `rmse` as auxiliary metrics.
- ADE20K semantic-segmentation validation applies matching letterbox geometry to images and masks, ignores source
  label `0`, maps labels `1..150` to model classes `0..149`, and reports mIoU as primary with pixel accuracy secondary.
- ImageNet validation reports Top-1 accuracy as the primary metric and Top-5 accuracy as the secondary metric.
- DOTAv1 validation reports rotated mAP50-95 as the primary metric and rotated mAP50 as the secondary metric.
- Preserve evaluator layouts. DOTAv1 stores its validation images directly in `images/` and may
  use `labels/val_original`, which retains difficult-object filtering.
- Expose a seed with default `0` for vision APIs, CLIs, benchmarks, and compatibility helpers that
  sample or otherwise use randomness.
- Keep qbcompiler imports inside the compilation path. Base imports, vision imports, the compile
  module import, and non-compile CLI commands must work without `qbcompiler` installed.
- Compilation accepts exactly one entry level: `data_path` (organize, sample, preprocess),
  `subset_path` (skip organization and sampling), or `calib_data_path` (validated `.npy` tensors;
  skip all preparation). Keep default models under `~/.mblt_model_zoo` and datasets under
  `~/.mblt_model_zoo/datasets`.

### Transformers and MeloTTS

- Install the matching optional extra before running integration tests.
- Start with the narrowest test file or documented `-k` selection. Use
  `pytest tests/transformers --full-matrix` only for a release or pre-merge matrix; use `-x` while
  iterating.
- Reuse shared NPU options and `tests.npu_backend_options.build_vision_engine_kwargs()` rather than
  adding divergent hardware flags or engine keyword bundles.
- Hardware, downloaded models, and external data may be unavailable. Run static or focused checks
  that are safe locally and state any limitation.

## Documentation and Validation

- Keep public documentation aligned with `MBLT_Engine`, `list_models()`, the shared
  `--model-path` option, and framework auto-detection.
- When a package update changes a durable fact—such as version support, dependencies or extras,
  public APIs, CLI commands, repository layout, validation, or workflow—update this guide and the
  applicable Codex and Claude skills in the same change. Reflect the change concisely in
  `CLAUDE.md` as well.
- Use ATX headings, one blank line between blocks, hyphen lists, language-tagged code fences, and
  concise paragraphs. Add `description` and `paths` YAML frontmatter to reusable agent rules or
  workflows.
- Follow the nearest guide before selecting a test:
  `tests/vision/TEST.md`, `tests/transformers/TEST.md`, `tests/MeloTTS/TEST.md`,
  `benchmark/vision/README.md`, or `benchmark/transformers/README.md`.
- Run the smallest meaningful validation. For a documentation-only change, check links, headings,
  and `git diff --check`; do not run hardware-bound suites unnecessarily.
- For Python changes, use the targeted relevant pytest file, then run
  `pre-commit run --files <touched files>` when available. Never bypass hooks with `--no-verify`.

## Git Safety

- Do not revert, format, or regenerate unrelated files.
- Keep commits focused and use Conventional Commit subjects under 50 characters in the imperative
  mood.
- Do not add generated artifacts, model weights, caches, or benchmark output unless the task
  explicitly requires them.
