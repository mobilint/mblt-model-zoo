---
name: mblt-vision
description: Work effectively on Mobilint Model Zoo vision models, datasets, evaluation, and compilation.
---

# Mobilint Model Zoo Vision

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-vision/SKILL.md`](../../../.agents/skills/mblt-vision/SKILL.md).

`eval_imagenet()` returns Top-1 as a float for compatibility, while `eval_imagenet_metrics()`
exposes structured Top-1 primary and Top-5 secondary metrics. DOTAv1 validation treats rotated
mAP50-95 as primary and rotated mAP50 as secondary.
Structured evaluator results expose primary and secondary score properties. Keep `eval_coco()`
numeric and use `eval_coco_metrics()` for structured mAP50-95 and mAP50.
The standard benchmark runner records NYU depth metrics and dispatches ADE20K or Cityscapes
semantic evaluation from `post_cfg.dataset`.
Auto-detect `.onnx` benchmark model paths before core-mode expansion and record one neutral `onnx`
target instead of repeated NPU-mode rows.

DOTAv1 loaders prefer flat `images/` validation files and accept legacy `images/val` as a fallback.
Local and archive organizers validate a complete staged DOTAv1 root before atomically replacing
the cache; successful installs remove stale files and failed rollbacks retain recoverable backups.

Model YAMLs derive same-stem ONNX Hub artifact names from `file_cfg.filename`; use
`onnx_filename` only for a non-matching artifact name.

Every model YAML declares `post_cfg.dataset`; dataset-aware postprocessing combines it with
`task` to resolve the output taxonomy and class count.

Use `obb` as the canonical vision task key for oriented bounding boxes and retain
`oriented_bounding_boxes` as a compatibility alias. Normalize the alias inside CLI, benchmark, and
DOTAv1 evaluator dispatch.

Reuse `vision.utils.letterbox.LetterBoxGeometry` for shared resize, padding, metadata, and inverse
crop calculations. Dense compilation resolves NYU Depth, ADE20K, or Cityscapes from
`post_cfg.dataset` and samples their organized RGB images.
Require `pre_cfg.LetterBox` for YOLO detection postprocessors. Semantic preprocessing metadata
contains original `img0_shape` and `ratio_pad`; prediction restores spatial logits before `argmax`,
and validation loaders reuse the same geometry for targets.

Normalize dense MXQ output before inverse letterboxing: depth `[1, H/4, W/4]` maps are bilinearly
upsampled by four, while Cityscapes semantic `[H, W, 19]` or `[B, H, W, 19]` logits become NCHW
before bilinear restoration and `argmax`. Keep existing ONNX layouts and baked maps compatible.
Define NYU Depth metric validity from targets and reject non-finite predictions at valid target pixels
before per-image median alignment.

ADE20K organization atomically installs its 2,000 flat validation pairs only after staged readiness
verifies `objectInfo150.txt` and `sceneCategories.txt` with `images/` and `annotations/`.
Reuse ImageNet, COCO, DOTAv1, WiderFace, and dense validation or calibration roots only after validating
taxonomy-specific layout, required metadata and targets, and the complete official split.
Validate complete staged ImageNet, COCO, and WiderFace roots before atomic replacement so failure
preserves the existing cache; require exact WiderFace event/image identity agreement with
`wider_face_val.mat`.
ADE20K semantic validation applies matching letterbox geometry to images and masks, pads masks with `255`, and reports
mIoU before pixel accuracy.
Cityscapes semantic validation uses only 500 paired validation images and `gtFine_labelIds` masks from the official
ZIP packages, the canonical 19-class source-ID mapping, and `255` ignore padding independently from ADE20K. Reject
other semantic validation taxonomies.
Reject semantic evaluator taxonomy overrides that differ from `model.post_cfg.dataset`.
