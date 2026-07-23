---
name: mblt-model-zoo
description: >-
  Work effectively in the Mobilint Model Zoo repository. Use for changes to vision models and
  datasets, optional qbcompiler compilation, Hugging Face Transformers integrations, MeloTTS, CLI
  commands, tests, benchmarks, or repository documentation that must respect Mobilint NPU and
  model-download constraints.
---

# Mobilint Model Zoo

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-model-zoo/SKILL.md`](../../../.agents/skills/mblt-model-zoo/SKILL.md).
Keep shared workflow content there so Codex and Claude Code stay synchronized.

For NYU Depth dataset organization, preserve only its 654 validation image/depth pairs as `images/` and `depth/`
directly under the output root.

For DOTAv1 dataset organization, preserve its 458 validation images directly under `images/` with both label layouts.
