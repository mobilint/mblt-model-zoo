---
name: mblt-model-zoo
description: >-
  Follow shared Mobilint Model Zoo workflow for repository-wide, CLI, MeloTTS, and documentation
  changes. Use the mblt-vision or mblt-transformers skill for area-specific work.
---

# Mobilint Model Zoo

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-model-zoo/SKILL.md`](../../../.agents/skills/mblt-model-zoo/SKILL.md).
Keep shared workflow content there so Codex and Claude Code stay synchronized.

Preserve model `post_cfg.dataset` metadata so vision output taxonomies are not inferred from task alone.
Require explicit YOLO LetterBox configuration. Semantic preprocessing exposes original `img0_shape`
and `ratio_pad`, and prediction restores spatial logits before `argmax`.

ADE20K organization atomically installs its 2,000 flat validation pairs only after staged readiness
verifies `objectInfo150.txt` and `sceneCategories.txt` with `images/` and `annotations/`.
ADE20K validation ignores source label `0`, maps labels `1..150` to classes `0..149`, and reports mIoU before pixel
accuracy.
Cityscapes organization retains only 500 paired validation images and `gtFine_labelIds` masks from the official ZIP
packages, maps its canonical 19 source IDs to train IDs, and never installs train or test data.
Semantic evaluators reject taxonomy overrides that differ from `model.post_cfg.dataset`.

Dense compilation uses `post_cfg.dataset` to select NYU Depth, ADE20K, or Cityscapes and samples
only their organized RGB images.
Validate baked semantic class IDs as finite, integral, and in-range before converting their dtype.
NYU Depth metric validity comes from targets; non-finite predictions at valid target pixels are rejected
before median alignment.
Reuse organized validation and calibration datasets only after validating taxonomy-specific layout,
required metadata and targets, and the complete official split.
Reject symlinked dense local source data and metadata, require resolved copy sources to remain
within the resolved dataset root, and reject symlinked dense managed roots during readiness.
Validate complete staged ImageNet, COCO, and WiderFace roots before atomic replacement so failure
preserves the existing cache; require exact WiderFace event/image identity agreement with
`wider_face_val.mat`.
Stage and validate local and archive DOTAv1 roots before atomic cache replacement; successful
installs remove stale files and failed rollbacks retain recoverable backups.
Normalize `oriented_bounding_boxes` to canonical `obb` at CLI, benchmark, and evaluator boundaries.
Keep TPS printed tables and JSON output synchronized through the shared
`mblt_model_zoo/cli/tps_table.py` schema.
Keep Vision evaluator primary and secondary score properties aligned; the standard benchmark
runner owns depth and taxonomy-dispatched semantic metrics.
Auto-detect `.onnx` benchmark model paths before core-mode expansion and record one neutral `onnx`
target instead of repeated NPU-mode rows.

Use the focused entry points for Vision and Transformers work:

- [`mblt-vision`](../mblt-vision/SKILL.md)
- [`mblt-transformers`](../mblt-transformers/SKILL.md)
