---
name: mblt-model-zoo
description: >-
  Work on the Mobilint Model Zoo package, its legacy compatibility facade, CLI integration,
  Transformers integrations, MeloTTS, and repository documentation. Use for Model Zoo changes;
  implement Vision features in the standalone mblt-vision-python repository instead.
---

# Mobilint Model Zoo

1. Read `AGENTS.md`, run `git status --short`, and inspect the relevant parser, exports, tests, and
   `pyproject.toml` before editing.
2. Keep `mblt_model_zoo.vision` and its Vision compilation exports as thin compatibility layers.
   Do not reintroduce Vision implementation, datasets, evaluation, benchmarks, or Vision tests.
3. Make new Vision CLI behavior in `mblt-vision-python` first. The Model Zoo `predict`, `val`, and
   `compile` handlers must delegate to `mblt_vision.cli`.
4. Preserve Model Zoo CLI help and README examples when its CLI integration changes. Pass
   board-specific `target_device` through to the standalone packages; do not reintroduce legacy
   product/artifact selection in Model Zoo.
5. Keep TPS output driven by `mblt_model_zoo/cli/tps_table.py`. Preserve local conventions in
   `hf_transformers` and `MeloTTS`.
6. Start with focused tests. Report unavailable hardware, downloads, and optional extras instead of
   weakening validation. For docs, run `git diff --check`.
7. When package ownership, public API, CLI bridges, or runtime dependencies change significantly,
   update `AGENTS.md`, this canonical skill, the Claude entry point when its workflow changes, and
   relevant documentation in the same change.
