---
name: mblt-model-zoo
description: >-
  Follow shared Mobilint Model Zoo workflow for repository-wide, CLI, MeloTTS, and documentation
  changes. Use the mblt-vision or mblt-transformers skill for area-specific work.
---

# Mobilint Model Zoo

## Start Here

1. Read `AGENTS.md`; it is the canonical shared agent guide and `CLAUDE.md` imports it.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml` and the nearest area guide before choosing dependencies or validation:
   - MeloTTS: `tests/MeloTTS/TEST.md` and `mblt_model_zoo/MeloTTS/README.md`.
4. For vision models, datasets, evaluation, or compilation, also use `mblt-vision`.
5. For `mblt_model_zoo/hf_transformers` or Transformers benchmarks, also use
   `mblt-transformers`.

## Shared Surface

- Use the installed `mblt-model-zoo` CLI for package behavior. Its native commands are `predict`,
  `val`, `compile`, `tps`, `melo`, and `melo-ui`; `classify`, `detect`, `pose`, `segment`, and
  `melotts` are aliases.
- When the CLI surface changes, synchronize parser `-h`/`--help` text and the README CLI guide.
  Verify the root help and every affected subcommand help output against the current interface.

## Preserve Shared Contracts

- When a package update changes a durable public fact or workflow, update `AGENTS.md`, this skill,
  the relevant area skill, `CLAUDE.md`, and the matching Claude skill entry point in the same change.
  Keep shared guidance concise.
- Preserve model `post_cfg.dataset` metadata so vision output taxonomies are not inferred from task
  alone.
- Atomically install ADE20K's 2,000 flat validation pairs only after staged readiness verifies
  `objectInfo150.txt` and `sceneCategories.txt` with the `images/` and `annotations/` directories.
- Preserve ADE20K's `0` ignore label and `1..150` to `0..149` validation mapping; report mIoU before pixel accuracy.
- Preserve Cityscapes's 500 validation-only lossless PNG pairs from the two official train/validation/test ZIP
  packages and its canonical 19-class source-ID mapping; never install train, test, or auxiliary annotation files.
- Reject semantic evaluator taxonomy overrides that differ from `model.post_cfg.dataset`.
- Normalize quarter-resolution MXQ depth and channel-last Cityscapes MXQ logits before inverse
  letterboxing while preserving full-resolution ONNX depth, NCHW logits, and baked class maps.
  Validate baked semantic IDs as finite, integral, and in-range before converting their dtype.
- Require explicit YOLO LetterBox configuration. Metadata-enabled semantic preprocessing exposes
  original shape and `ratio_pad`; prediction restores spatial logits before `argmax`.
- Route dense compilation through `post_cfg.dataset`: NYU Depth for depth models and ADE20K or
  Cityscapes for semantic models.
- Define NYU Depth metric validity from targets and reject non-finite predictions at valid target pixels
  before median alignment.
- Reuse organized validation and calibration datasets only after checking their taxonomy-specific
  layout, required metadata and targets, and full official split count.
- Validate complete staged ImageNet, COCO, and WiderFace roots before atomic replacement so failure
  preserves the existing cache; validate WiderFace event/image identities against `wider_face_val.mat`.
- Stage and validate complete local or archive DOTAv1 roots before atomic replacement. Remove stale
  managed files on success and preserve recoverable backups when rollback fails.
- Normalize `oriented_bounding_boxes` to canonical `obb` at CLI, benchmark, and evaluator boundaries.
- Keep `mblt-model-zoo tps` printed tables and JSON output driven by the shared
  `mblt_model_zoo/cli/tps_table.py` schema.
- Keep Vision evaluator results aligned on primary and secondary score properties. The standard
  benchmark runner owns depth and taxonomy-dispatched semantic metrics.
- Auto-detect `.onnx` benchmark model paths before core-mode expansion and record one neutral
  `onnx` target instead of repeated NPU-mode rows.
- Do not force formatting standards on `hf_transformers` or `MeloTTS`; follow local style.

## Validate Proportionately

- Run one targeted test file or `-k` selection first; append `-x` while fixing failures.
- Expect NPU tests and model-download tests to be environment-dependent. Report unavailable
  hardware, artifacts, or extras rather than broadening the test run.
- Run `pre-commit run --files <touched files>` for Python edits when hooks are available. For docs,
  run `git diff --check` and inspect rendered Markdown structure.

## Avoid

- Do not revert unrelated user work, commit generated outputs, or bypass pre-commit hooks.
