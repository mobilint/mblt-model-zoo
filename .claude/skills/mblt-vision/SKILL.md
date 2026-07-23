---
name: mblt-vision
description: Work effectively on Mobilint Model Zoo vision models, datasets, evaluation, and compilation.
---

# Mobilint Model Zoo Vision

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-vision/SKILL.md`](../../../.agents/skills/mblt-vision/SKILL.md).

ImageNet validation treats Top-1 accuracy as primary and Top-5 accuracy as secondary. DOTAv1
validation treats rotated mAP50-95 as primary and rotated mAP50 as secondary.

Model YAMLs derive same-stem ONNX Hub artifact names from `file_cfg.filename`; use
`onnx_filename` only for a non-matching artifact name.

Every model YAML declares `post_cfg.dataset`; dataset-aware postprocessing combines it with
`task` to resolve the output taxonomy and class count.

ADE20K organization preserves its 2,000 validation image/mask pairs as flat `images/` and `annotations/`
directories.
