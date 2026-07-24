# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical repository guidance. When a durable workflow changes, update
  it alongside this guide and both Codex and Claude skills.
- Keep this guide synchronized with `AGENTS.md` for shared repository guidance so Claude Code and
  Codex receive the same workflow requirements.
- The Claude Code shared-skill entry point is `.claude/skills/mblt-model-zoo/SKILL.md`; its shared
  content is maintained in `.agents/skills/mblt-model-zoo/SKILL.md`. Use the focused Vision and
  Transformers entries in `.claude/skills/mblt-vision/` and `.claude/skills/mblt-transformers/`.
- When the CLI changes, keep parser `-h`/`--help` text and the README CLI guide synchronized; check
  the root help and each affected subcommand help output.
- NYU Depth organization installs only its 654 validation image/depth pairs as `images/` and `depth/` at the output root.
- NYU Depth evaluation uses stretched inputs, per-image median alignment, and pooled valid-pixel statistics; `delta1`
  is its primary score, with `abs_rel` and `rmse` also reported.
- ImageNet validation uses Top-1 accuracy as its primary metric and Top-5 accuracy as its secondary metric.
- DOTAv1 validation uses rotated mAP50-95 as its primary metric and rotated mAP50 as its secondary metric.
- DOTAv1 organization installs its 458 validation images directly under `images/` and retains both label layouts.
- Model YAMLs use `file_cfg.filename` as the canonical MXQ artifact; the matching same-stem ONNX
  artifact is derived unless `onnx_filename` explicitly names an exception.
- Every model YAML declares `post_cfg.dataset`; postprocessing uses it with `task` to resolve the
  model's output taxonomy and class count.
- Preserve anchorless decoded-output layout provenance through NMS. If provenance is unavailable
  for an ambiguous tensor, normalize it as raw channels-first before candidates-first.
- ADE20K organization installs its 2,000 validation image/mask pairs as flat `images/` and `annotations/` directories.
- ADE20K semantic validation uses matched letterbox geometry, ignores source label `0`, and reports mIoU followed by
  pixel accuracy.
- Cityscapes organization installs only 500 validation PNG pairs from `Chris1/cityscapes`; validation maps the 19
  canonical source IDs to train IDs and ignores all other labels.
- Read the nearest area README or `TEST.md` before modifying code or selecting validation.
- Preserve unrelated working-tree changes and report environment-dependent test limitations.
