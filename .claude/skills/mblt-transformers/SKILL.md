---
name: mblt-transformers
description: Work effectively on Mobilint Model Zoo Hugging Face Transformers integrations and benchmarks.
---

# Mobilint Model Zoo Transformers

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-transformers/SKILL.md`](../../../.agents/skills/mblt-transformers/SKILL.md).

Treat `mblt_model_zoo/cli/tps_table.py` as the source of truth for TPS printed rows and JSON output.
Keep VLM non-batch tests under `image_text_to_text/non_batch`; matrix-runner Phase B owns both batch
text-generation and batch image-text-to-text suites.
