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
Require explicit YOLO LetterBox configuration and metadata-enabled semantic validation preprocessing.

ADE20K organization preserves its 2,000 validation image/mask pairs as flat `images/` and `annotations/`
directories.
ADE20K validation ignores source label `0`, maps labels `1..150` to classes `0..149`, and reports mIoU before pixel
accuracy.
Cityscapes organization retains only 500 paired validation images and `gtFine_labelIds` masks from the official ZIP
packages, maps its canonical 19 source IDs to train IDs, and never installs train or test data.

Dense compilation uses `post_cfg.dataset` to select NYU Depth, ADE20K, or Cityscapes and samples
only their organized RGB images.
NYU Depth metric validity comes from targets; non-finite predictions at valid target pixels are rejected
before median alignment.
Reuse organized validation and calibration datasets only after validating taxonomy-specific layout,
required metadata and targets, and the complete official split.
Keep TPS printed tables and JSON output synchronized through the shared
`mblt_model_zoo/cli/tps_table.py` schema.

Use the focused entry points for Vision and Transformers work:

- [`mblt-vision`](../mblt-vision/SKILL.md)
- [`mblt-transformers`](../mblt-transformers/SKILL.md)
