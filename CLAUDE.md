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
- NYU Depth evaluation uses stretched inputs, per-image median alignment, and pooled target-valid-pixel statistics;
  it rejects non-finite predictions at those pixels and reports `delta1`, `abs_rel`, and `rmse`.
- `eval_imagenet()` returns Top-1 as a float for compatibility; `eval_imagenet_metrics()` exposes
  structured Top-1 primary and Top-5 secondary metrics.
- Evaluator result objects expose `primary_score` and `secondary_score`; `eval_coco()` remains
  numeric while `eval_coco_metrics()` exposes structured mAP50-95 and mAP50.
- The standard Vision benchmark runner records NYU depth metrics and dispatches ADE20K or
  Cityscapes semantic metrics from `post_cfg.dataset`.
- Auto-detect `.onnx` benchmark `--model-path` values before core-mode expansion and record one
  neutral `onnx` target for `--core-mode all`.
- DOTAv1 validation uses rotated mAP50-95 as its primary metric and rotated mAP50 as its secondary metric.
- DOTAv1 organization installs its 458 validation images directly under `images/` and retains both label layouts;
  its loader also accepts legacy `images/val` datasets. Local and archive organizers validate a
  complete staged root before atomic replacement, remove stale files on success, and preserve a
  recoverable backup if rollback fails.
- Model YAMLs use `file_cfg.filename` as the canonical MXQ artifact; the matching same-stem ONNX
  artifact is derived unless `onnx_filename` explicitly names an exception.
- Every model YAML declares `post_cfg.dataset`; postprocessing uses it with `task` to resolve the
  model's output taxonomy and class count.
- Use `obb` as the canonical vision task key for oriented bounding boxes; retain
  `oriented_bounding_boxes` as a compatibility alias and normalize it at CLI, benchmark, and
  evaluator boundaries.
- Preserve anchorless decoded-output layout provenance through NMS. If provenance is unavailable
  for an ambiguous tensor, normalize it as raw channels-first before candidates-first.
- Reuse `vision.utils.letterbox.LetterBoxGeometry` for shared forward and inverse letterbox
  geometry.
- Require `pre_cfg.LetterBox` for YOLO detection postprocessing, and use metadata-enabled semantic
  preprocessing that exposes original `img0_shape` and `ratio_pad`. Prediction restores spatial
  logits with this geometry before `argmax`, while validation loaders reuse it for targets.
- Normalize dense MXQ outputs before inverse letterboxing: upsample depth `[1, H/4, W/4]` maps by
  four, and convert Cityscapes `[H, W, 19]` or `[B, H, W, 19]` logits to NCHW before restoration
  and `argmax`. Preserve existing ONNX layouts and baked class maps; validate baked IDs as finite,
  integral, and in-range before converting to `int64`.
- Resolve dense compilation datasets from `post_cfg.dataset`: NYU Depth, ADE20K, or Cityscapes;
  sample calibration inputs from their organized `images/` directories.
- ADE20K organization atomically installs its 2,000 validation pairs with required
  `objectInfo150.txt` and `sceneCategories.txt` metadata after validating the complete staged root.
- Dense local organizers reject symlinked data and metadata and require resolved copy sources to
  remain within the resolved dataset root; readiness rejects symlinked dense managed roots.
- ADE20K semantic validation uses matched letterbox geometry, ignores source label `0`, and reports mIoU followed by
  pixel accuracy.
- Cityscapes organization selects only 500 validation PNG pairs from the two official train/validation/test ZIP
  packages; validation maps the 19 canonical source IDs to train IDs and ignores all other labels. Reject semantic
  validation taxonomies other than `ade20k` and `cityscapes`, including evaluator overrides that
  differ from `model.post_cfg.dataset`.
- Reuse ImageNet, COCO, DOTAv1, WiderFace, and dense validation or calibration roots only after
  checking taxonomy-specific layout, required metadata and targets, and the complete official split.
- Validate complete staged ImageNet, COCO, and WiderFace roots before atomic replacement, preserving
  an existing cache on failure; match WiderFace event/image identities exactly to `wider_face_val.mat`.
- Keep TPS table and JSON output synchronized through `mblt_model_zoo/cli/tps_table.py`.
- Keep VLM non-batch tests under `image_text_to_text/non_batch`; matrix-runner Phase B owns the
  batch text-generation and image-text-to-text suites.
- Read the nearest area README or `TEST.md` before modifying code or selecting validation.
- Preserve unrelated working-tree changes and report environment-dependent test limitations.
