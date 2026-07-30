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

DOTAv1 loaders prefer flat `images/` validation files and accept legacy `images/val` as a fallback.

Model YAMLs derive same-stem ONNX Hub artifact names from `file_cfg.filename`; use
`onnx_filename` only for a non-matching artifact name.

Every model YAML declares `post_cfg.dataset`; dataset-aware postprocessing combines it with
`task` to resolve the output taxonomy and class count.

Use `obb` consistently as the vision task key for oriented bounding boxes.

Reuse `vision.utils.letterbox.LetterBoxGeometry` for shared resize, padding, metadata, and inverse
crop calculations. Dense compilation resolves NYU Depth, ADE20K, or Cityscapes from
`post_cfg.dataset` and samples their organized RGB images.

Normalize dense MXQ output before inverse letterboxing: depth `[1, H/4, W/4]` maps are bilinearly
upsampled by four, while Cityscapes semantic `[H, W, 19]` or `[B, H, W, 19]` logits become NCHW
before bilinear restoration and `argmax`. Keep existing ONNX layouts and baked maps compatible.

ADE20K organization preserves its 2,000 validation image/mask pairs as flat `images/` and `annotations/`
directories.
ADE20K semantic validation applies matching letterbox geometry to images and masks, pads masks with `255`, and reports
mIoU before pixel accuracy.
Cityscapes semantic validation uses only 500 paired validation images and `gtFine_labelIds` masks from the official
ZIP packages, the canonical 19-class source-ID mapping, and `255` ignore padding independently from ADE20K. Reject
other semantic validation taxonomies.
