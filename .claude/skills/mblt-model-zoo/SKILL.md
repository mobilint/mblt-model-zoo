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

ADE20K organization preserves its 2,000 validation image/mask pairs as flat `images/` and `annotations/`
directories.
ADE20K validation ignores source label `0`, maps labels `1..150` to classes `0..149`, and reports mIoU before pixel
accuracy.

Use the focused entry points for Vision and Transformers work:

- [`mblt-vision`](../mblt-vision/SKILL.md)
- [`mblt-transformers`](../mblt-transformers/SKILL.md)
